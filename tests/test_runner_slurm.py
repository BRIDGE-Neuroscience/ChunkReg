"""The SLURM runner: what it writes, what it submits, what it reports.

None of this needs a scheduler. Script generation is exercised through
``dry_run``, and submission and polling are exercised against two stub
executables that stand in for ``sbatch`` and ``sacct``, which also pins the
argument lists the real ones would receive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chunkreg.config import Resources, RunConfig, SlurmConfig, SubjectSpec, get_profile
from chunkreg.runners import get_runner
from chunkreg.runners.base import RunReport, TaskStatus
from chunkreg.runners.slurm import SlurmRunner, _parse_sacct

PASSES = ("register", "blend", "update")


def make_cfg(tmp_path: Path, **slurm_kw) -> RunConfig:
    return RunConfig(
        root=str(tmp_path / "run"),
        subjects=(SubjectSpec("s01", "subjects/s01.zarr"),),
        profile=get_profile("a16"),
        profile_name="a16",
        runner="slurm",
        slurm=SlurmConfig(**slurm_kw) if slurm_kw else SlurmConfig(),
    )


@pytest.fixture
def cfg(tmp_path) -> RunConfig:
    return make_cfg(tmp_path)


@pytest.fixture
def runner(tmp_path, cfg) -> SlurmRunner:
    """A dry-run runner bound to a level and iteration, as the driver binds it."""
    r = SlurmRunner(cfg, dry_run=True)
    return r.bind(tmp_path / "config.yaml", level=1, iteration=2)


def script_text(runner: SlurmRunner, pass_name: str, ids=range(4)) -> str:
    return runner.write_script(pass_name, list(ids)).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Stub scheduler
# --------------------------------------------------------------------------- #
SBATCH_STUB = '''\
import pathlib, sys

pathlib.Path(__file__).with_name("sbatch_argv.txt").write_text(
    "\\n".join(sys.argv[1:]), encoding="utf-8"
)
print("12345;cluster0")
'''

SACCT_STUB = '''\
import pathlib, sys

pathlib.Path(__file__).with_name("sacct_argv.txt").write_text(
    "\\n".join(sys.argv[1:]), encoding="utf-8"
)
print({rows!r})
'''

ROWS = "\n".join(
    [
        "12345_0|COMPLETED",
        "12345_0.batch|COMPLETED",
        "12345_0.extern|COMPLETED",
        "12345_1|FAILED",
        "12345_1.batch|CANCELLED",
        "12345_2|COMPLETED",
        "12345_2.extern|COMPLETED",
        "12345_3|OUT_OF_MEMORY",
        "12345_3.batch|OUT_OF_MEMORY",
    ]
)


def make_scheduler(tmp_path, cfg, rows=ROWS):
    """A runner pointed at stub sbatch/sacct executables."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "sbatch.py").write_text(SBATCH_STUB, encoding="utf-8")
    (bin_dir / "sacct.py").write_text(SACCT_STUB.format(rows=rows), encoding="utf-8")
    r = SlurmRunner(
        cfg,
        poll_seconds=0.0,
        max_wait_s=10.0,
        submit_cmd=[sys.executable, str(bin_dir / "sbatch.py")],
        sacct_cmd=[sys.executable, str(bin_dir / "sacct.py")],
    )
    r.bind(tmp_path / "config.yaml", level=0, iteration=0)
    return r, bin_dir


@pytest.fixture
def scheduler(tmp_path, cfg):
    return make_scheduler(tmp_path, cfg)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_the_registry_builds_a_slurm_runner():
    r = get_runner("slurm", dry_run=True)
    assert isinstance(r, SlurmRunner)
    assert r.name == "slurm"


# --------------------------------------------------------------------------- #
# Script generation
# --------------------------------------------------------------------------- #
def test_dry_run_writes_a_script_per_pass_and_submits_nothing(runner, cfg):
    reports = {p: runner.run(p, lambda tid: tid, range(4)) for p in PASSES}

    for pass_name, report in reports.items():
        assert report.ok
        assert [s.task_id for s in report.statuses] == [0, 1, 2, 3]
        script = cfg.root_path / "slurm" / f"{pass_name}_L1_it2.sbatch"
        assert script.is_file()
    # Nothing was submitted, so there is no job to chain onto.
    assert runner.last_job_id is None


def test_a_dry_run_needs_no_scheduler_at_all(tmp_path, cfg):
    """The point of dry_run: it must not care that sbatch is missing."""
    r = SlurmRunner(cfg, dry_run=True, submit_cmd="no-such-sbatch-on-this-box")
    r.bind(tmp_path / "config.yaml", 0, 0)
    assert r.run("blend", None, range(3)).ok


def test_only_the_register_script_asks_for_a_gpu(runner):
    assert "#SBATCH --gres=gpu:1" in script_text(runner, "register")
    assert "#SBATCH --gres" not in script_text(runner, "blend")
    assert "#SBATCH --gres" not in script_text(runner, "update")


def test_the_account_directive_is_omitted_when_no_account_is_set(runner):
    for pass_name in PASSES:
        text = script_text(runner, pass_name)
        assert "--account" not in text
        # The omission must not leave a hole in the directive block.
        assert "\n\n#SBATCH" not in text


def test_the_account_directive_appears_when_one_is_configured(tmp_path):
    cfg = make_cfg(tmp_path, account="proj-123")
    r = SlurmRunner(cfg, dry_run=True).bind(tmp_path / "config.yaml", 1, 2)
    assert "#SBATCH --account=proj-123" in script_text(r, "register")


def test_the_array_range_matches_the_number_of_ids(runner):
    assert "#SBATCH --array=0-6\n" in script_text(runner, "blend", ids=range(7))
    assert "#SBATCH --array=0-0\n" in script_text(runner, "blend", ids=range(1))


def test_a_throttle_limits_how_many_elements_run_at_once(tmp_path, cfg):
    r = SlurmRunner(cfg, dry_run=True, throttle=8)
    r.bind(tmp_path / "config.yaml", 1, 2)
    assert "#SBATCH --array=0-19%8\n" in script_text(r, "register", ids=range(20))


def test_a_sparse_retry_list_keeps_its_own_task_ids(runner):
    """Array index is the task id, so a retry lands on the tasks that failed."""
    assert runner.array_spec([3, 9, 4]) == "3,4,9"


def test_resources_and_paths_reach_the_directives(runner, cfg, tmp_path):
    text = script_text(runner, "register")
    res = cfg.slurm.register
    assert "#SBATCH --job-name=register_L1_it2" in text
    assert f"#SBATCH --partition={cfg.slurm.partition}" in text
    assert f"#SBATCH --cpus-per-task={res.cpus}" in text
    assert f"#SBATCH --mem={res.mem_gb}G" in text
    assert f"#SBATCH --time={res.time}" in text
    assert "logs/register_L1_it2_%a.out" in text
    assert "logs/register_L1_it2_%a.err" in text


def test_the_element_invokes_run_task_with_its_own_address(runner, tmp_path):
    text = script_text(runner, "update")
    cfg_path = (tmp_path / "config.yaml").as_posix()
    assert f'chunkreg run-task "{cfg_path}"' in text
    assert "--level 1" in text
    assert "--iter 2" in text
    assert "--pass update" in text


def test_shell_variables_survive_template_substitution(runner):
    """string.Template must not eat the one value only the scheduler knows."""
    text = script_text(runner, "register")
    assert "--id $SLURM_ARRAY_TASK_ID" in text
    assert "$SLURM_JOB_ID" in text
    assert "$(hostname)" in text
    assert "set -euo pipefail" in text
    # Nothing that looks like an unfilled placeholder may remain.
    assert "${" not in text


def test_an_unknown_pass_falls_back_to_the_blend_resources(tmp_path):
    own = Resources(gpus=0, cpus=2, mem_gb=7, time="00:10:00")
    cfg = make_cfg(tmp_path, update=own)
    r = SlurmRunner(cfg, dry_run=True).bind(tmp_path / "config.yaml", 0, 0)
    assert r.resources("qc") == cfg.slurm.blend
    assert r.template_path("qc").name == "blend.sbatch"
    # The named passes still get their own block, not the fallback.
    assert r.resources("update").mem_gb == 7


def test_an_empty_id_list_submits_nothing(runner, cfg):
    report = runner.run("register", None, [])
    assert report.ok and report.statuses == []
    assert not (cfg.root_path / "slurm" / "register_L1_it2.sbatch").exists()


def test_running_unbound_says_what_is_missing(cfg):
    with pytest.raises(RuntimeError, match="bind"):
        SlurmRunner(cfg, dry_run=True).run("register", None, range(2))
    with pytest.raises(RuntimeError, match="RunConfig"):
        SlurmRunner(dry_run=True, cfg_path="c.yaml").run("register", None, range(2))


# --------------------------------------------------------------------------- #
# Submission and polling
# --------------------------------------------------------------------------- #
def test_sacct_rows_become_one_status_per_array_element(scheduler):
    runner, _ = scheduler
    report = runner.run("register", None, range(4))

    assert runner.last_job_id == "12345"  # the ";cluster" suffix is not part of it
    assert [(s.task_id, s.ok) for s in report.statuses] == [
        (0, True),
        (1, False),
        (2, True),
        (3, False),
    ]
    assert not report.ok
    assert all(s.error is None for s in report.statuses if s.ok)


def test_a_failure_carries_its_state_and_the_log_to_read(scheduler):
    runner, _ = scheduler
    report = runner.run("register", None, range(4))

    oom = report.statuses[3]
    assert "OUT_OF_MEMORY" in oom.error
    assert str(runner.log_path("register", 3)) in oom.error
    assert oom.error.endswith("register_L0_it0_3.out")
    # The sub-steps are ignored, so element 1 reports FAILED and not the
    # CANCELLED of its .batch step.
    assert "FAILED" in report.statuses[1].error


def test_failed_ids_lists_exactly_what_a_retry_resubmits(scheduler):
    runner, _ = scheduler
    report = runner.run("register", None, range(4))
    assert runner.failed_ids(report) == [1, 3]
    assert runner.failed_ids(RunReport("blend", [TaskStatus(7, True)])) == []


def test_the_dependency_chains_the_next_pass_after_this_one(scheduler):
    runner, bin_dir = scheduler
    runner.run("register", None, range(4))
    first = runner.last_job_id

    runner.dependency = first
    runner.run("blend", None, range(4))
    argv = (bin_dir / "sbatch_argv.txt").read_text(encoding="utf-8").splitlines()

    assert "--parsable" in argv
    assert f"--dependency=afterok:{first}" in argv
    assert argv[-1].endswith("blend_L0_it0.sbatch")


def test_the_first_pass_is_submitted_without_a_dependency(scheduler):
    runner, bin_dir = scheduler
    runner.run("register", None, range(4))
    argv = (bin_dir / "sbatch_argv.txt").read_text(encoding="utf-8").splitlines()
    assert not any(a.startswith("--dependency") for a in argv)


def test_sacct_is_asked_for_the_parsable_columns_the_parser_reads(scheduler):
    runner, bin_dir = scheduler
    runner.run("blend", None, range(4))
    argv = (bin_dir / "sacct_argv.txt").read_text(encoding="utf-8").splitlines()
    assert argv == [
        "-j",
        "12345",
        "--format=JobID,State",
        "--noheader",
        "--parsable2",
    ]


def test_a_terminal_state_nobody_recognises_is_still_a_failure(tmp_path, cfg):
    """Better a reported oddity than a poll loop that never ends."""
    runner, _ = make_scheduler(tmp_path, cfg, rows="12345_0|SPECIAL_EXIT")
    report = runner.run("blend", None, range(1))
    assert not report.ok
    assert "SPECIAL_EXIT" in report.statuses[0].error
    assert "unrecognised" in report.statuses[0].error


def test_an_element_sacct_never_mentions_is_not_reported_as_success(tmp_path, cfg):
    runner, _ = make_scheduler(
        tmp_path, cfg, rows="12345_0|COMPLETED\n12345_1|COMPLETED"
    )
    # sacct knows about 0 and 1, so the wait ends, but 2 was asked for as well.
    report = runner._report("blend", [0, 1, 2], runner.poll("12345"), "12345")
    assert [s.ok for s in report.statuses] == [True, True, False]
    assert "no state reported by sacct" in report.statuses[2].error


def test_a_job_that_never_finishes_is_given_up_on_rather_than_polled_forever(
    tmp_path, cfg
):
    runner, _ = make_scheduler(tmp_path, cfg, rows="12345_0|PENDING")
    runner.max_wait_s = 0.0
    with pytest.raises(RuntimeError, match="sacct -j 12345"):
        runner.run("blend", None, range(1))


def test_a_missing_sbatch_is_an_actionable_error(tmp_path, cfg):
    r = SlurmRunner(cfg, submit_cmd="chunkreg-no-such-sbatch")
    r.bind(tmp_path / "config.yaml", 0, 0)
    with pytest.raises(RuntimeError, match="not found on PATH") as exc:
        r.run("register", None, range(2))
    message = str(exc.value)
    assert "chunkreg-no-such-sbatch" in message
    assert "dry_run" in message and "login" in message


# --------------------------------------------------------------------------- #
# sacct parsing
# --------------------------------------------------------------------------- #
def test_sub_steps_and_the_job_row_are_not_elements():
    """A step's state can disagree with its element's; only the element counts."""
    states = _parse_sacct(
        "12345_0|COMPLETED\n"
        "12345_0.batch|FAILED\n"
        "12345_0.extern|COMPLETED\n"
        "12345|RUNNING\n"
        "\n"
    )
    assert states == {0: "COMPLETED"}


def test_an_unexpanded_range_still_carries_its_state():
    """An array cancelled before it expands has only its range row.

    Dropping that row left the poll loop waiting for terminal states that were
    never going to arrive, so an array cancelled while pending hung the driver.
    """
    states = _parse_sacct("12345_[0-3]|CANCELLED by 1001\n")
    assert states == {i: "CANCELLED" for i in range(4)}


def test_a_concrete_row_beats_the_range_it_came_from():
    states = _parse_sacct(
        "12345_[0-3]|PENDING\n"
        "12345_0|COMPLETED\n"
        "12345_1|FAILED\n"
    )
    assert states == {0: "COMPLETED", 1: "FAILED", 2: "PENDING", 3: "PENDING"}


def test_a_throttled_or_listed_range_expands():
    assert set(_parse_sacct("9_[0-2%2]|PENDING\n")) == {0, 1, 2}
    assert set(_parse_sacct("9_[1,4,6-7]|PENDING\n")) == {1, 4, 6, 7}
    assert _parse_sacct("9_[bad]|PENDING\n") == {}


def test_a_cancellation_reports_the_state_without_the_user_id():
    assert _parse_sacct("7_2|CANCELLED by 1001\n7_3|TIMEOUT+\n") == {
        2: "CANCELLED",
        3: "TIMEOUT",
    }
