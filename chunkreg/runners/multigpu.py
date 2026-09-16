"""Run a pass across every GPU of one machine, one worker process per GPU.

For a job that holds several GPUs under one scheduler allocation, such as a
Grid Engine job with ``-pe gpu 6``. The driver process only coordinates, seeds
level 0 and writes small metadata; each worker owns one GPU, loads the feature
network and the registration engine once, and then takes tasks from a shared
queue until the run ends. Those tasks are the register, blend and update
passes and also promotion and settling, one chunk core per task, so no step of
a run is left waiting on a single process while the other GPUs idle. A worker is pinned to its GPU with ``CUDA_VISIBLE_DEVICES`` before
PyTorch starts, so inside it the GPU is simply ``cuda``.

Like the SLURM runner, this one never runs the callable it is handed: a
closure over the manifest cannot be sent to another process. It sends the
task's address instead, ``(pass, level, iteration, task id)``, and the worker
rebuilds the work from the same config file and the manifest the driver wrote.
Unlike the SLURM runner it gets each task's result back, so the stopping rule
sees the records directly.

Workers are started with ``spawn``, never ``fork``: the driver may already hold
a CUDA context, and a forked child cannot use it.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from itertools import count
from pathlib import Path
from typing import Any, Callable, Sequence

from .base import RunReport, TaskStatus

__all__ = ["MultiGPURunner", "visible_gpus"]


def visible_gpus() -> list[str]:
    """The GPU ids this process may use, as ``CUDA_VISIBLE_DEVICES`` spells them.

    A scheduler that hands a job some of a node's GPUs says which through that
    variable, so its entries, not ``0..n-1``, are what each worker is pinned to.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip():
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        import torch
    except ImportError:
        return []
    return [str(i) for i in range(torch.cuda.device_count())]


def _worker(
    slot: int,
    gpu: str | None,
    config_path: str,
    device: str,
    engine_name: str,
    inbox,
    outbox,
) -> None:
    """One GPU's loop. Runs in its own process."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CHUNKREG_DEVICE"] = device
    try:
        from .. import xp
        from ..config import load_config
        from ..engines import get_engine
        from ..features import get_extractor
        from ..passes import run_blend_task, run_register_task, run_update_task
        from ..passes.manifest import Manifest
        from ..passes.promote import carry_task

        cfg = load_config(config_path)
        dev = xp.configure(device)
        engine = get_engine(engine_name)
        extractor = get_extractor(cfg.profile.features)
        extractor.setup()
    except BaseException:  # noqa: BLE001 - reported to the driver
        outbox.put(("dead", slot, None, traceback.format_exc()))
        return
    outbox.put(("ready", slot, None, f"gpu {gpu} as {dev}"))

    manifests: dict[tuple[int, int], Any] = {}
    while True:
        msg = inbox.get()
        if msg is None:
            break
        job, pass_name, level, iteration, task_id, params = msg
        outbox.put(("start", slot, (job, task_id), None))
        try:
            if pass_name == "carry":
                out = carry_task(cfg, params["spec"], task_id)
                outbox.put(("done", slot, (job, task_id), out))
                continue
            key = (level, iteration)
            manifest = manifests.get(key)
            if manifest is None:
                if len(manifests) > 4:
                    manifests.clear()
                path = cfg.level_dir(level) / f"manifest_it{iteration}.json"
                manifest = manifests[key] = Manifest.load(path)
            if pass_name == "register":
                out = run_register_task(
                    cfg, manifest, task_id, engine=engine, extractor=extractor
                )
            elif pass_name == "blend":
                out = run_blend_task(cfg, manifest, task_id)
            elif pass_name == "update":
                out = run_update_task(cfg, manifest, task_id)
            else:
                raise ValueError(f"unknown pass {pass_name!r}")
            outbox.put(("done", slot, (job, task_id), out))
        except Exception:  # noqa: BLE001 - reported to the driver
            outbox.put(("fail", slot, (job, task_id), traceback.format_exc()))
        finally:
            # The driver shares the first GPU (it seeds level 0 there), so a
            # worker must not sit on memory it is not using.
            xp.release()


class MultiGPURunner:
    """Dispatch task ids to one persistent worker per GPU."""

    name = "multigpu"
    runs_carry = True
    """Promotion and settling run as per-core tasks on the workers too."""

    def __init__(
        self,
        cfg=None,
        *,
        gpus: Sequence[str] | None = None,
        engine: str | None = None,
        device: str = "cuda",
        config_path: str | Path | None = None,
        poll_seconds: float = 5.0,
    ) -> None:
        self.cfg = cfg
        self.gpus = list(gpus) if gpus is not None else visible_gpus()
        if not self.gpus:
            raise RuntimeError("no GPUs are visible to this process")
        self.engine = engine or (cfg.engine_for(device) if cfg is not None else "fireants")
        self.device = device
        self.config_path = None if config_path is None else str(config_path)
        self.poll_seconds = float(poll_seconds)
        self.level = 0
        self.iteration = 0
        self._ctx = mp.get_context("spawn")
        self._inbox = None
        self._outbox = None
        self._procs: list = []
        self._jobs = count()
        self._deaths: dict[int, str] = {}
        self._idle_polls = 0

    # -- protocol ----------------------------------------------------------- #
    def bind(self, config_path, level: int, iteration: int, cfg=None) -> "MultiGPURunner":
        if config_path is None:
            raise ValueError("the multi-GPU runner needs the run's config file path")
        path = str(config_path)
        if self.config_path is not None and path != self.config_path and self._procs:
            raise ValueError("workers were started for a different config file")
        self.config_path = path
        self.level = int(level)
        self.iteration = int(iteration)
        if cfg is not None:
            self.cfg = cfg
        return self

    def run(
        self,
        pass_name: str,
        fn: Callable[..., Any],
        ids: Sequence[int],
        **kwargs: Any,
    ) -> RunReport:
        report = RunReport(pass_name=pass_name)
        ids = list(ids)
        if not ids:
            return report
        self._start()
        from .. import xp

        xp.release()  # whatever the driver cached while promoting
        job = next(self._jobs)
        params = dict(kwargs)
        for i in ids:
            self._inbox.put((job, pass_name, self.level, self.iteration, int(i), params))

        pending = set(ids)
        running: dict[int, int] = {}  # slot -> task id
        results: dict[int, TaskStatus] = {}
        self._idle_polls = 0
        while pending:
            try:
                kind, slot, key, payload = self._outbox.get(timeout=self.poll_seconds)
            except queue.Empty:
                self._reap(running, pending, results, pass_name)
                continue
            self._idle_polls = 0
            if kind in ("ready",):
                continue
            if kind == "dead":
                self._deaths[slot] = str(payload)
                self._fail_dead(slot, payload, running, pending, results)
                continue
            got_job, task_id = key
            if got_job != job:
                continue  # left over from an abandoned pass
            if kind == "start":
                running[slot] = task_id
            elif kind in ("done", "fail"):
                running.pop(slot, None)
                if task_id in pending:
                    pending.discard(task_id)
                    results[task_id] = TaskStatus(
                        task_id=task_id,
                        ok=kind == "done",
                        result=payload if kind == "done" else None,
                        error=None if kind == "done" else payload,
                    )
        report.statuses = [results[i] for i in sorted(results)]
        return report

    def close(self) -> None:
        if not self._procs:
            return
        for _ in self._procs:
            try:
                self._inbox.put(None)
            except (OSError, ValueError):
                pass
        deadline = time.monotonic() + 30
        for p in self._procs:
            p.join(timeout=max(0.1, deadline - time.monotonic()))
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        self._procs = []

    def __enter__(self) -> "MultiGPURunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- workers ------------------------------------------------------------ #
    def _start(self) -> None:
        if self._procs:
            return
        if self.config_path is None:
            raise RuntimeError(
                "the multi-GPU runner was never bound to a config file; call "
                "build_template with config_path=..."
            )
        self._inbox = self._ctx.Queue()
        self._outbox = self._ctx.Queue()
        for slot, gpu in enumerate(self.gpus):
            p = self._ctx.Process(
                target=_worker,
                args=(
                    slot, gpu, self.config_path, self.device, self.engine,
                    self._inbox, self._outbox,
                ),
                name=f"chunkreg-gpu{gpu}",
                daemon=True,
            )
            p.start()
            self._procs.append(p)

    def _reap(self, running, pending, results, pass_name) -> None:
        """Account for workers that died without reporting."""
        alive = 0
        for slot, p in enumerate(self._procs):
            if p.is_alive():
                alive += 1
                continue
            self._fail_dead(
                slot,
                f"worker for GPU {self.gpus[slot]} exited with code {p.exitcode}",
                running,
                pending,
                results,
            )
        # A worker killed between taking a task and announcing it leaves that
        # task nowhere: not queued, not running. Live workers that are all idle
        # with an empty queue for several polls in a row mean exactly that.
        # A worker announces a task as soon as it takes one, so a few polls
        # cannot mistake a slow start for a lost task.
        if alive and pending and not running and self._inbox.empty():
            self._idle_polls += 1
        else:
            self._idle_polls = 0
        if self._idle_polls >= 3:
            for task_id in sorted(pending):
                results[task_id] = TaskStatus(
                    task_id=task_id,
                    ok=False,
                    error=(
                        f"{pass_name} task was taken by a GPU worker that died "
                        f"before starting it; rerun to retry"
                    ),
                )
            pending.clear()
            return
        if alive == 0 and pending:
            why = "\n".join(
                f"GPU {self.gpus[s]}: {msg}" for s, msg in sorted(self._deaths.items())
            )
            for task_id in sorted(pending):
                results[task_id] = TaskStatus(
                    task_id=task_id,
                    ok=False,
                    error=(
                        f"every GPU worker has exited; {pass_name} task never ran"
                        + (f"\n{why}" if why else "")
                    ),
                )
            pending.clear()

    @staticmethod
    def _fail_dead(slot, why, running, pending, results) -> None:
        task_id = running.pop(slot, None)
        if task_id is not None and task_id in pending:
            pending.discard(task_id)
            results[task_id] = TaskStatus(task_id=task_id, ok=False, error=str(why))
