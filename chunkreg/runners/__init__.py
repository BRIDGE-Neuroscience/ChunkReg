"""Runners: where task ids execute.

A runner is handed a pass name and a list of task ids and is responsible only
for getting each id run somewhere. It never decides what a task does. That
separation is why the same pass functions serve a laptop with one GPU and a
cluster with a thousand.
"""

from __future__ import annotations

from .base import Runner, TaskStatus
from .local import LocalRunner

__all__ = [
    "Runner",
    "TaskStatus",
    "LocalRunner",
    "get_runner",
    "gpu_slots",
    "runner_for",
]


def __getattr__(name):
    if name == "MultiGPURunner":
        from .multigpu import MultiGPURunner

        return MultiGPURunner
    raise AttributeError(name)


def get_runner(name: str, **kwargs) -> Runner:
    if name == "local":
        return LocalRunner(**kwargs)
    if name == "slurm":
        from .slurm import SlurmRunner

        return SlurmRunner(**kwargs)
    if name == "multigpu":
        from .multigpu import MultiGPURunner

        return MultiGPURunner(**kwargs)
    raise ValueError(f"unknown runner {name!r}; use 'local' or 'slurm'")


def gpu_slots(cfg, device: str, say=None) -> list[str]:
    """The worker slots a local run spreads over, as GPU ids.

    One entry is one worker process. A GPU appears once per worker it is to
    host, so ``workers_per_gpu`` above one simply repeats it: the runner pins
    each worker to the id it was handed, and two workers pinned to the same id
    share that card.

    Only a plain ``cuda`` device spreads: ``cuda:N`` asks for one card, and a
    CPU device has none.
    """
    say = say or (lambda _msg: None)
    if device != "cuda":
        return []
    from .multigpu import visible_gpus

    ids = visible_gpus()
    if cfg.gpus != "all":
        want = int(cfg.gpus)
        if want > len(ids):
            say(
                f"  warning: gpus is {want} but this job can see {len(ids)} GPU(s) "
                f"({', '.join(ids) or 'none'}); using those"
            )
        ids = ids[:want]

    per, why = cfg.workers_on_each_gpu()
    if per > 1 and ids:
        say(f"  {per} workers per GPU ({why})")
        ids = [g for g in ids for _ in range(per)]
    return ids


def runner_for(cfg, device: str, engine_name: str, config_path, workers: int = 1,
               say=None):
    """The runner a run dispatches through, and the engine the driver holds.

    ``config_path`` must name the config the *run* is under, which for a
    pairwise run is the derived pair config rather than the cohort's: a worker
    process and a batch node each rebuild their task by loading that file.

    The engine comes back only for an in-process runner. Anything that spreads
    over other processes builds its own there, since a CUDA context and a
    feature network do not survive a fork.
    """
    say = say or (lambda _msg: None)
    from ..engines import get_engine

    if cfg.runner == "slurm":
        return get_runner(
            "slurm",
            cfg=cfg,
            poll_seconds=cfg.slurm.poll_seconds,
            max_wait_s=cfg.slurm.max_wait_s,
        ), None

    gpus = gpu_slots(cfg, device, say=say)
    if len(gpus) > 1:
        cards = sorted(set(gpus), key=gpus.index)
        say(
            f"  running {len(gpus)} worker(s) on {len(cards)} GPU(s) "
            f"({', '.join(cards)})"
        )
        # Bound to the config file up front: a resumed run can promote a level
        # on the workers before it runs any pass.
        return get_runner(
            "multigpu", cfg=cfg, gpus=gpus, engine=engine_name, device="cuda",
            config_path=str(config_path),
        ), None

    if device.startswith("cuda") and workers > 1:
        # Threads would share one card, and one registration can already fill
        # it. Parallelism on a GPU comes from more GPUs.
        say(
            f"  note: workers={workers} ignored on a single GPU; "
            f"tasks run one at a time"
        )
        workers = 1
    return LocalRunner(workers=workers), get_engine(engine_name)
