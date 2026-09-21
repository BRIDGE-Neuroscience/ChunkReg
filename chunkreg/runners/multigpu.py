"""Run a pass across every GPU of one machine, one worker process per GPU.

For a job that holds several GPUs under one scheduler allocation, such as a
Grid Engine job with ``-pe gpu 6``. The driver process only coordinates, seeds
level 0 and writes small metadata; each worker owns one GPU, loads the feature
network and the registration engine once, and then takes tasks from a shared
queue until the run ends. Those tasks are the register, blend and update
passes and also promotion and settling, one chunk core per task, so no step of
a run is left waiting on a single process while the other GPUs idle. A worker is pinned to its GPU with ``CUDA_VISIBLE_DEVICES`` before
PyTorch starts, so inside it the GPU is simply ``cuda``.

Passes are not always run one at a time. A blend task depends on a known
handful of register tasks rather than on all of them, so the driver can hand
the workers both passes at once with that dependency attached and let each
core's blend start the moment its own chunks are done. What it removes is the
barrier at the end of every register pass, where a node's GPUs go idle one by
one waiting for the slowest task.

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
from typing import Any, Callable, Iterable, Mapping, Sequence

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
        outbox.put(("start", slot, (job, pass_name, task_id), None))
        try:
            if pass_name == "carry":
                out = carry_task(cfg, params["spec"], task_id)
                outbox.put(("done", slot, (job, pass_name, task_id), out))
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
            outbox.put(("done", slot, (job, pass_name, task_id), out))
        except Exception:  # noqa: BLE001 - reported to the driver
            outbox.put(("fail", slot, (job, pass_name, task_id), traceback.format_exc()))
        finally:
            # The driver shares the first GPU (it seeds level 0 there), so a
            # worker must not sit on memory it is not using.
            xp.release()


class MultiGPURunner:
    """Dispatch task ids to one persistent worker per GPU."""

    name = "multigpu"
    runs_carry = True
    """Promotion and settling run as per-core tasks on the workers too."""
    runs_fused = True
    """Register and blend can run as one dependency-ordered dispatch."""

    def __init__(
        self,
        cfg=None,
        *,
        gpus: Sequence[str] | None = None,
        engine: str | None = None,
        device: str = "cuda",
        config_path: str | Path | None = None,
        poll_seconds: float = 5.0,
        progress_seconds: float = 60.0,
    ) -> None:
        self.cfg = cfg
        self.gpus = list(gpus) if gpus is not None else visible_gpus()
        if not self.gpus:
            raise RuntimeError("no GPUs are visible to this process")
        self.engine = engine or (cfg.engine_for(device) if cfg is not None else "fireants")
        self.device = device
        self.config_path = None if config_path is None else str(config_path)
        self.poll_seconds = float(poll_seconds)
        self.progress_seconds = float(progress_seconds)
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
        return self._dispatch([(pass_name, list(ids))], {}, kwargs)[pass_name]

    def run_chain(
        self,
        first: str,
        first_ids: Sequence[int],
        second: str,
        second_ids: Sequence[int],
        depends: Mapping[int, Iterable[int]],
        **kwargs: Any,
    ) -> tuple[RunReport, RunReport]:
        """Run two passes at once, each item of the second gated on the first.

        ``depends`` maps an id of ``second`` to the ids of ``first`` it reads.
        An item of the second pass is queued the moment the last of its
        dependencies reports, so the workers move on to it while the rest of
        the first pass is still running instead of standing idle at a barrier
        that only a few of them are still holding up.

        A dependency that fails abandons everything downstream of it rather
        than letting a task read a result that was never written; the abandoned
        items come back as failures naming the one that failed.
        """
        blocked = {
            (second, int(i)): {(first, int(d)) for d in deps}
            for i, deps in depends.items()
        }
        reports = self._dispatch(
            [(first, list(first_ids)), (second, list(second_ids))], blocked, kwargs
        )
        return reports[first], reports[second]

    # -- dispatch ----------------------------------------------------------- #
    def _dispatch(
        self,
        stages: Sequence[tuple[str, Sequence[int]]],
        blocked_on: Mapping[tuple[str, int], set[tuple[str, int]]],
        params: dict,
    ) -> dict[str, RunReport]:
        """Run several passes' items over the workers, honouring dependencies.

        Work is addressed as ``(pass name, id)`` throughout, because with two
        passes in flight an id alone no longer identifies a task.
        """
        order: dict[str, list[int]] = {}
        items: list[tuple[str, int]] = []
        for name, ids in stages:
            order[name] = [int(i) for i in ids]
            items += [(name, int(i)) for i in order[name]]
        reports = {name: RunReport(pass_name=name) for name, _ in stages}
        if not items:
            return reports

        self._start()
        from .. import xp

        xp.release()  # whatever the driver cached while promoting
        job = next(self._jobs)

        pending = set(items)
        waiting = {it: set(d) for it, d in blocked_on.items() if it in pending and d}
        waiters: dict[tuple[str, int], list[tuple[str, int]]] = {}
        for it, deps in waiting.items():
            for d in deps:
                waiters.setdefault(d, []).append(it)
        results: dict[tuple[str, int], TaskStatus] = {}
        running: dict[int, tuple[str, int]] = {}

        # A pass at a chunked level is hundreds of tasks over several minutes,
        # and with the engine's own chatter silenced the driver was the only
        # thing that could say the run was alive. One throttled line beats
        # either extreme.
        started = time.monotonic()
        last_said = [started]

        def tick(force: bool = False) -> None:
            if self.progress_seconds <= 0:
                return
            now = time.monotonic()
            if not force and now - last_said[0] < self.progress_seconds:
                return
            last_said[0] = now
            total = len(items)
            failed = sum(1 for s in results.values() if not s.ok)
            print(
                f"    {'+'.join(order)}: {total - len(pending)}/{total} done"
                + (f", {failed} failed" if failed else "")
                + f", {len(running)} running, {now - started:.0f}s",
                flush=True,
            )

        def send(item: tuple[str, int]) -> None:
            name, task_id = item
            self._inbox.put((job, name, self.level, self.iteration, task_id, params))

        def record(item: tuple[str, int], ok: bool, payload: Any) -> None:
            pending.discard(item)
            waiting.pop(item, None)
            results[item] = TaskStatus(
                task_id=item[1],
                ok=ok,
                result=payload if ok else None,
                error=None if ok else str(payload),
            )

        def release(item: tuple[str, int], ok: bool) -> None:
            """Queue, or abandon, whatever was waiting on a finished item."""
            stack = [(item, ok)]
            while stack:
                done, done_ok = stack.pop()
                for w in waiters.pop(done, ()):
                    if w not in waiting:
                        continue
                    if not done_ok:
                        record(
                            w,
                            False,
                            f"{w[0]} task {w[1]} never ran: it reads the result "
                            f"of {done[0]} task {done[1]}, which failed",
                        )
                        stack.append((w, False))
                        continue
                    deps = waiting[w]
                    deps.discard(done)
                    if not deps:
                        waiting.pop(w)
                        send(w)

        def fail_dead(slot: int, why: Any) -> None:
            item = running.pop(slot, None)
            if item is not None and item in pending:
                record(item, False, why)
                release(item, False)

        def reap() -> None:
            """Account for workers that died without reporting."""
            alive = 0
            for slot, proc in enumerate(self._procs):
                if proc.is_alive():
                    alive += 1
                    continue
                fail_dead(
                    slot,
                    f"worker for GPU {self.gpus[slot]} exited with code "
                    f"{proc.exitcode}",
                )
            # A worker killed between taking a task and announcing it leaves
            # that task nowhere: not queued, not running. Live workers that are
            # all idle with an empty queue for several polls in a row mean
            # exactly that. A worker announces a task as soon as it takes one,
            # so a few polls cannot mistake a slow start for a lost task.
            if alive and pending and not running and self._inbox.empty():
                self._idle_polls += 1
            else:
                self._idle_polls = 0
            if self._idle_polls >= 3:
                for item in sorted(pending):
                    record(
                        item,
                        False,
                        f"{item[0]} task {item[1]} was taken by a GPU worker "
                        f"that died before starting it; rerun to retry"
                        if item not in waiting
                        else f"{item[0]} task {item[1]} never ran: the pass "
                        f"stalled with its dependencies unfinished",
                    )
                return
            if alive == 0 and pending:
                why = "\n".join(
                    f"GPU {self.gpus[s]}: {msg}"
                    for s, msg in sorted(self._deaths.items())
                )
                for item in sorted(pending):
                    record(
                        item,
                        False,
                        f"every GPU worker has exited; {item[0]} task "
                        f"{item[1]} never ran" + (f"\n{why}" if why else ""),
                    )

        for item in items:
            if item not in waiting:
                send(item)

        self._idle_polls = 0
        while pending:
            try:
                kind, slot, key, payload = self._outbox.get(timeout=self.poll_seconds)
            except queue.Empty:
                reap()
                tick()
                continue
            self._idle_polls = 0
            if kind == "ready":
                continue
            if kind == "dead":
                self._deaths[slot] = str(payload)
                fail_dead(slot, payload)
                continue
            got_job, got_pass, task_id = key
            if got_job != job:
                continue  # left over from an abandoned pass
            item = (got_pass, int(task_id))
            if kind == "start":
                running[slot] = item
            elif kind in ("done", "fail"):
                running.pop(slot, None)
                if item in pending:
                    record(item, kind == "done", payload)
                    release(item, kind == "done")
                tick()

        for name, ids in order.items():
            reports[name].statuses = [
                results[(name, i)] for i in ids if (name, i) in results
            ]
        return reports

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

