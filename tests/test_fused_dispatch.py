"""Register and blend as one dependency-ordered dispatch.

The barrier between the two passes is what leaves most of a node's GPUs idle
behind the last few register tasks of every pass. Removing it means a blend
task has to know exactly which register tasks it reads, and a runner has to
honour that -- including when one of them fails, where the cost of getting it
wrong is a blend reading a task array that was never written.

The dispatch logic is exercised against stub worker queues rather than real
processes, so the ordering and failure rules are tested directly instead of
inferred from timings.
"""

from __future__ import annotations

import queue
import threading

import pytest

from chunkreg.config import (
    LevelPolicy,
    Resources,
    RunConfig,
    SlurmConfig,
    StageSpec,
    SubjectSpec,
)
from chunkreg.grid import GridSpec, Profile, pyramid
from chunkreg.passes import build_manifest
from chunkreg.passes.blend import blend_depends
from chunkreg.runners.multigpu import MultiGPURunner


@pytest.fixture
def manifest_cfg(small_profile):
    cfg = RunConfig(
        root=".",
        subjects=(
            SubjectSpec("s01", "subjects/s01.zarr"),
            SubjectSpec("s02", "subjects/s02.zarr"),
        ),
        profile=small_profile,
        profile_name="test",
        backend="memory",
        gpu_mem_gb=1000.0,
        levels=LevelPolicy(
            caps=(1, 1, 1),
            seeded_stages=(StageSpec(kind="greedy", scales=(1,), iterations=(2,)),),
        ),
        slurm=SlurmConfig(register=Resources(gpus=1, chunks_per_task=2)),
    )
    # Four cores per axis, so an interior chunk exists whose read set is a
    # small fraction of the level rather than all of it.
    grid = pyramid(GridSpec((128, 128, 128), 0.05), small_profile)[-1]
    return cfg, build_manifest(cfg, 2, 0, grid)


# --------------------------------------------------------------------------- #
# The dependency map
# --------------------------------------------------------------------------- #
def test_blend_dependencies_are_exactly_the_read_set(manifest_cfg):
    """Derived from the chunk lattice, but identical to scanning every chunk."""
    cfg, m = manifest_cfg
    deps = blend_depends(cfg, m)
    assert set(deps) == {c.id for c in m.chunks}
    for chunk in m.chunks:
        expected = {
            m.locate(s, other.id)[0]
            for other in m.chunks_touching_core(chunk.id)
            for s in m.subjects
        }
        assert deps[chunk.id] == expected


def test_a_blend_does_not_wait_on_the_whole_register_pass(manifest_cfg):
    """The point of the map: a core reads its neighbours, not the level."""
    cfg, m = manifest_cfg
    deps = blend_depends(cfg, m)
    assert len(m.chunks) > 8, "test needs a grid with an interior"
    assert m.n_tasks > 1, "test needs more than one register task"
    interior = max(m.chunks, key=lambda c: min(c.index))
    assert 0 < len(deps[interior.id]) < m.n_tasks
    # Every chunk reads itself, whatever else it reads.
    for chunk in m.chunks:
        assert m.locate(m.subjects[0], chunk.id)[0] in deps[chunk.id]


# --------------------------------------------------------------------------- #
# The runner honouring it
# --------------------------------------------------------------------------- #
class _FakeProc:
    exitcode = None

    def is_alive(self) -> bool:
        return True

    def join(self, timeout=None) -> None:
        pass


def _stub_runner() -> MultiGPURunner:
    runner = MultiGPURunner(
        gpus=["0"], engine="demons", device="cpu", config_path="run.json",
        poll_seconds=0.05,
    )
    runner._inbox = queue.Queue()
    runner._outbox = queue.Queue()
    runner._procs = [_FakeProc()]
    return runner


def _pump(runner, fails=()):
    """Answer every dispatched task, recording what was known done when it was.

    Returns the thread and the log, one entry per task taken off the queue:
    ``(pass name, id, frozenset of tasks already reported done)``.
    """
    log = []
    done: set[tuple[str, int]] = set()

    def loop():
        while True:
            msg = runner._inbox.get()
            if msg is None:
                return
            job, name, _level, _iteration, task_id, _params = msg
            log.append((name, task_id, frozenset(done)))
            key = (job, name, task_id)
            runner._outbox.put(("start", 0, key, None))
            if (name, task_id) in fails:
                runner._outbox.put(("fail", 0, key, "boom"))
            else:
                done.add((name, task_id))
                runner._outbox.put(("done", 0, key, {"id": task_id}))

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread, log


def test_a_blend_is_queued_only_once_every_chunk_it_reads_is_done(manifest_cfg):
    cfg, m = manifest_cfg
    deps = blend_depends(cfg, m)
    runner = _stub_runner()
    thread, log = _pump(runner)
    try:
        reg, bl = runner.run_chain(
            "register", range(m.n_tasks), "blend", range(len(m.chunks)), deps
        )
    finally:
        runner._inbox.put(None)
        thread.join(timeout=5)

    assert reg.ok and bl.ok
    assert len(reg.statuses) == m.n_tasks
    assert len(bl.statuses) == len(m.chunks)
    taken = {(name, task_id) for name, task_id, _ in log}
    assert taken == {("register", i) for i in range(m.n_tasks)} | {
        ("blend", c.id) for c in m.chunks
    }
    for name, task_id, already_done in log:
        if name != "blend":
            continue
        missing = {("register", d) for d in deps[task_id]} - already_done
        assert not missing, f"blend {task_id} started before {sorted(missing)}"


def test_a_failed_register_abandons_only_the_blends_that_read_it(manifest_cfg):
    cfg, m = manifest_cfg
    deps = blend_depends(cfg, m)
    broken = 0
    poisoned = {cid for cid, on in deps.items() if broken in on}
    assert poisoned and len(poisoned) < len(m.chunks), "test needs both kinds"

    runner = _stub_runner()
    thread, log = _pump(runner, fails={("register", broken)})
    try:
        reg, bl = runner.run_chain(
            "register", range(m.n_tasks), "blend", range(len(m.chunks)), deps
        )
    finally:
        runner._inbox.put(None)
        thread.join(timeout=5)

    assert [s.task_id for s in reg.failed] == [broken]
    assert {s.task_id for s in bl.failed} == poisoned
    assert "register task 0" in bl.failed[0].error
    # A blend that never ran must never have been handed to a worker either.
    started = {task_id for name, task_id, _ in log if name == "blend"}
    assert started == {c.id for c in m.chunks} - poisoned


def test_one_pass_on_its_own_still_works(manifest_cfg):
    """``run`` is the same dispatch with a single stage and no dependencies."""
    _cfg, m = manifest_cfg
    runner = _stub_runner()
    thread, log = _pump(runner)
    try:
        report = runner.run("register", None, range(m.n_tasks))
    finally:
        runner._inbox.put(None)
        thread.join(timeout=5)
    assert report.ok
    assert [s.task_id for s in report.statuses] == list(range(m.n_tasks))
    assert [name for name, _, _ in log] == ["register"] * m.n_tasks
