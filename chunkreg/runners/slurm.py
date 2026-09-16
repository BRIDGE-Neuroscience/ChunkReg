"""Run one pass as an sbatch array, one array element per task id.

Why this runner ignores the callable it is handed
=================================================
``Runner.run`` receives the body of a pass as a Python callable, and the
pipeline builds that callable as a closure over the manifest, the engine and
the feature extractor. A closure cannot be shipped to a batch node: there is
nothing to pickle, and nothing the node could import to rebuild it. So this
runner never touches ``fn``. It ships an *address* instead. Every task in the
system is already addressable as (config, level, iteration, pass, task id),
which is exactly the argument list of ``chunkreg run-task``, so an array
element is one line of shell and the node reconstructs the work from the same
config file the driver read.

That address is why :meth:`SlurmRunner.bind` exists. The runner protocol gives
``run`` only the pass name and the ids; the level and iteration are not
recoverable from those, so the driver binds them once per pass before calling.

What the caller gets back
=========================
A :class:`RunReport` with one :class:`TaskStatus` per array element, built from
``sacct`` rather than from return values. A batch node leaves its results in
the store, not in this process, so ``TaskStatus.result`` is always ``None``,
and a failure carries the SLURM state together with the path to that element's
log, which is the only place its traceback exists.

Chaining and retries
====================
The passes of one template pass are strictly ordered, so each submission can
depend on the previous one with ``afterok`` instead of the driver blocking on
``sacct`` between them; :attr:`last_job_id` is what the caller chains on.
:meth:`failed_ids` turns a finished report back into the id list a retry
submits, which is what ``chunkreg status --retry`` resubmits.
"""

from __future__ import annotations

import re
import shutil
import string
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config import Resources, RunConfig
from .base import RunReport, TaskStatus

__all__ = ["SlurmRunner"]


# The three passes that own a resource block and a template. Anything else is
# CPU work shaped like blend, so blend is the fallback rather than an error: a
# new pass should be runnable before it is special-cased here.
_KNOWN_PASSES = ("register", "blend", "update")
_FALLBACK_PASS = "blend"

_OK_STATE = "COMPLETED"

_FAILED_STATES = frozenset(
    {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
)
"""The states an operator expects to see on a bad element. Not the only states
treated as failure; see :data:`_ACTIVE_STATES`."""

_ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "REQUEUED",
        "REQUEUE_HOLD",
        "RESIZING",
        "SUSPENDED",
        "COMPLETING",
        "CONFIGURING",
    }
)
"""States that mean "ask again later". Everything outside this set is terminal,
including states this module has never heard of: polling forever on an unknown
state would hang a run that has already finished, whereas reporting it as a
failure hands the operator a state name and a log path to look at."""


class _ScriptTemplate(string.Template):
    """A template in which only the braced form is a placeholder.

    The scripts are full of shell variables, and ``$SLURM_ARRAY_TASK_ID`` is
    the one piece of an array element that only the scheduler can supply, so
    the default :class:`string.Template` is unusable here: it would claim every
    bare ``$NAME`` and either substitute it or reject the script. The pattern
    below matches ``${name}`` only, and makes the ``escaped``, ``named`` and
    ``invalid`` groups unmatchable, so anything else beginning with ``$`` is
    not a placeholder at all and is copied through exactly as written.
    """

    pattern = r"""
        \$(?:
            (?P<escaped>(?!))                        |
            \{(?P<braced>[_a-zA-Z][_a-zA-Z0-9]*)\}   |
            (?P<named>(?!))                          |
            (?P<invalid>(?!))
        )
    """


def _render(template_path: Path, keys: Mapping[str, Any]) -> str:
    """Fill in one sbatch template, reporting an unknown key against the file."""
    raw = template_path.read_text(encoding="utf-8")
    try:
        text = _ScriptTemplate(raw).substitute(keys)
    except KeyError as exc:
        raise KeyError(
            f"{template_path} uses the placeholder {exc.args[0]!r}, which "
            f"SlurmRunner does not provide. Known placeholders: "
            f"{', '.join(sorted(keys))}."
        ) from exc
    # An omitted optional directive leaves a blank line behind in the middle of
    # the #SBATCH block. sbatch tolerates it, but the script is also read by
    # people, and a hand-written one would not have the gap.
    return re.sub(r"\n[ \t]*\n(?=#SBATCH)", "\n", text)


def _parse_sacct(text: str) -> dict[int, str]:
    """Read ``JobID,State`` rows from ``sacct --parsable2`` into {index: state}.

    Every array element shows up several times: once as ``<job>_<index>`` and
    again as ``<job>_<index>.batch`` and ``.extern``. Only the element itself
    carries the state the report needs, and the sub-steps can disagree with it
    (a step killed for memory is not recorded the way its element is), so rows
    with a dot in the JobID are dropped. Elements not yet expanded appear as a
    range such as ``<job>_[4-9]`` and are dropped too: they have no state of
    their own, and their absence is what keeps the poll loop waiting.
    """
    states: dict[int, str] = {}
    for line in text.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split("|")
        if len(parts) < 2:
            continue
        job_id, state = parts[0].strip(), parts[1].strip()
        if "." in job_id or "_" not in job_id:
            continue
        index = job_id.split("_", 1)[1]
        if not index.isdigit():
            continue
        # "CANCELLED by 1001", and the "+" sacct appends to a truncated state,
        # are the same state as far as the report is concerned.
        states[int(index)] = state.split()[0].rstrip("+") if state else ""
    return states


def _argv(cmd: str | Sequence[str]) -> list[str]:
    """Accept "sbatch" or an argv prefix such as ["ssh", "login1", "sbatch"]."""
    if isinstance(cmd, (str, Path)):
        return [str(cmd)]
    return [str(part) for part in cmd]


class SlurmRunner:
    """Submit each pass as one sbatch array and wait for it.

    The runner is bound to a (config path, level, iteration) before each pass
    because the array elements need that address to rebuild their own work.
    ``dry_run`` writes the scripts and reports success without submitting,
    which is how the generation path is tested and how an operator inspects
    what would be sent before sending it.
    """

    name = "slurm"

    def __init__(
        self,
        cfg: RunConfig | None = None,
        *,
        cfg_path: str | Path | None = None,
        level: int = 0,
        iteration: int = 0,
        throttle: int | None = None,
        dependency: str | None = None,
        poll_seconds: float = 30.0,
        max_wait_s: float | None = None,
        dry_run: bool = False,
        submit_cmd: str | Sequence[str] = "sbatch",
        sacct_cmd: str | Sequence[str] = "sacct",
        template_dir: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.cfg_path = Path(cfg_path) if cfg_path is not None else None
        self.level = int(level)
        self.iteration = int(iteration)
        self.throttle = None if throttle is None else int(throttle)
        self.dependency = dependency
        # A 30 second poll costs one sacct call per element-minute of a job
        # that runs for hours. Tests drop it to milliseconds against a stub.
        self.poll_seconds = float(poll_seconds)
        self.max_wait_s = None if max_wait_s is None else float(max_wait_s)
        self.dry_run = bool(dry_run)
        self.submit_cmd = _argv(submit_cmd)
        self.sacct_cmd = _argv(sacct_cmd)
        self.template_dir = Path(template_dir) if template_dir is not None else None
        self.last_job_id: str | None = None

    # -- binding ------------------------------------------------------------ #
    def bind(
        self,
        cfg_path: str | Path,
        level: int,
        iteration: int,
        cfg: RunConfig | None = None,
    ) -> "SlurmRunner":
        """Address the next pass. Returns self so it can be called inline."""
        self.cfg_path = Path(cfg_path)
        self.level = int(level)
        self.iteration = int(iteration)
        if cfg is not None:
            self.cfg = cfg
        return self

    # -- the runner protocol ------------------------------------------------ #
    def run(
        self,
        pass_name: str,
        fn: Callable[..., Any] | None,
        ids: Sequence[int],
        **kwargs: Any,
    ) -> RunReport:
        """Submit ``ids`` as an array and block until every element is terminal.

        ``fn`` is accepted and discarded. It is a closure over objects that
        exist only in this process, so it cannot be sent to a batch node; the
        array elements re-derive the same work from the bound config instead.
        See the module docstring.
        """
        del fn  # deliberately unused; see the docstring above
        dependency = kwargs.pop("dependency", self.dependency)

        task_ids = [int(i) for i in ids]
        if not task_ids:
            # Nothing to submit is not an error, and an empty array is a
            # submission sbatch rejects.
            return RunReport(pass_name=pass_name)

        script = self.write_script(pass_name, task_ids)
        if self.dry_run:
            self.last_job_id = None
            return RunReport(
                pass_name=pass_name,
                statuses=[TaskStatus(task_id=i, ok=True) for i in task_ids],
            )

        job_id = self.submit(script, dependency=dependency)
        self.last_job_id = job_id
        states = self._wait(job_id, task_ids)
        return self._report(pass_name, task_ids, states, job_id)

    def failed_ids(self, report: RunReport) -> list[int]:
        """The ids to resubmit, in submission order. Used by ``status --retry``."""
        return [s.task_id for s in report.failed]

    # -- script generation -------------------------------------------------- #
    def script_stem(self, pass_name: str) -> str:
        """Job name, script name and log prefix are one string by design.

        An operator who sees a job name in ``squeue`` can then find the script
        and the logs of that exact submission without consulting anything.
        """
        return f"{pass_name}_L{self.level}_it{self.iteration}"

    def log_path(self, pass_name: str, task_id: int, stream: str = "out") -> Path:
        """Where one array element's output lands: ``%a`` resolved for an id."""
        cfg = self._require_cfg()
        stem = self.script_stem(pass_name)
        return cfg.root_path / "logs" / f"{stem}_{task_id}.{stream}"

    def write_script(self, pass_name: str, ids: Sequence[int]) -> Path:
        """Render the template for one pass into ``<root>/slurm``."""
        cfg = self._require_cfg()
        cfg_path = self._require_cfg_path()
        res = self.resources(pass_name)
        task_ids = [int(i) for i in ids]

        script_dir = cfg.root_path / "slurm"
        log_dir = cfg.root_path / "logs"
        script_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        stem = self.script_stem(pass_name)
        account = cfg.slurm.account
        keys: dict[str, Any] = {
            "job_name": stem,
            "partition": cfg.slurm.partition,
            # The directive is dropped rather than left empty: on a cluster
            # with a default account, "--account=" is a rejection, while no
            # --account at all is what leaving it unset was meant to say.
            "account_line": f"#SBATCH --account={account}" if account else "",
            "array": self.array_spec(task_ids),
            "gpus": res.gpus,
            "cpus": res.cpus,
            "mem_gb": res.mem_gb,
            "time": res.time,
            # POSIX spellings whatever the driver runs on: the script is bash,
            # and a Windows driver would otherwise turn a perfectly good
            # cluster path into one full of backslashes.
            "log_out": (log_dir / f"{stem}_%a.out").as_posix(),
            "log_err": (log_dir / f"{stem}_%a.err").as_posix(),
            "config": cfg_path.as_posix(),
            "level": self.level,
            "iteration": self.iteration,
            "pass_name": pass_name,
        }

        path = script_dir / f"{stem}.sbatch"
        text = _render(self.template_path(pass_name), keys)
        # Written with explicit LF: a driver on Windows would otherwise emit
        # CRLF, and bash on the batch node fails on the carriage returns.
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return path

    def resources(self, pass_name: str) -> Resources:
        cfg = self._require_cfg()
        which = pass_name if pass_name in _KNOWN_PASSES else _FALLBACK_PASS
        return getattr(cfg.slurm, which)

    def template_path(self, pass_name: str) -> Path:
        name = pass_name if pass_name in _KNOWN_PASSES else _FALLBACK_PASS
        return self._template_dir() / f"{name}.sbatch"

    def array_spec(self, ids: Sequence[int]) -> str:
        """The ``--array`` value for these ids, throttle included.

        Array indices are task ids, never positions. That is what lets the
        script hand ``$SLURM_ARRAY_TASK_ID`` straight to ``run-task``, and it
        is what makes a retry of a sparse id list address the tasks that
        failed rather than renumbering them.
        """
        task_ids = [int(i) for i in ids]
        if task_ids == list(range(len(task_ids))):
            spec = f"0-{len(task_ids) - 1}"
        else:
            spec = ",".join(str(i) for i in sorted(set(task_ids)))
        if self.throttle:
            spec += f"%{self.throttle}"
        return spec

    # -- submission and polling --------------------------------------------- #
    def submit(self, script: Path, dependency: str | None = None) -> str:
        """Submit one script and return its job id."""
        argv = list(self.submit_cmd) + ["--parsable"]
        if dependency:
            # afterok, not afterany: a blend that runs over a half-written set
            # of task arrays produces a silently wrong template.
            argv.append(f"--dependency=afterok:{dependency}")
        argv.append(str(script))

        out = self._check_output(argv, role="sbatch")
        head = out.strip().splitlines()[0].strip() if out.strip() else ""
        # --parsable prints "jobid" locally and "jobid;cluster" on a federation.
        job_id = head.split(";")[0].strip()
        if not job_id:
            raise RuntimeError(
                f"sbatch accepted {script} but printed no job id; its output "
                f"was {out!r}. SlurmRunner needs the --parsable job id both to "
                f"poll this pass and to chain the next one."
            )
        return job_id

    def poll(self, job_id: str) -> dict[int, str]:
        """Current state of every expanded element of an array job."""
        argv = list(self.sacct_cmd) + [
            "-j",
            str(job_id),
            "--format=JobID,State",
            "--noheader",
            "--parsable2",
        ]
        return _parse_sacct(self._check_output(argv, role="sacct"))

    def _wait(self, job_id: str, ids: Sequence[int]) -> dict[int, str]:
        started = time.monotonic()
        while True:
            states = self.poll(job_id)
            if all(i in states and states[i] not in _ACTIVE_STATES for i in ids):
                return states
            if (
                self.max_wait_s is not None
                and time.monotonic() - started > self.max_wait_s
            ):
                stuck = [i for i in ids if states.get(i, "") in _ACTIVE_STATES]
                raise RuntimeError(
                    f"job {job_id} has not finished after {self.max_wait_s:.0f}s; "
                    f"{len(stuck)} of {len(ids)} elements are still active. "
                    f"Inspect it with: sacct -j {job_id}"
                )
            time.sleep(self.poll_seconds)

    def _report(
        self,
        pass_name: str,
        ids: Sequence[int],
        states: Mapping[int, str],
        job_id: str,
    ) -> RunReport:
        report = RunReport(pass_name=pass_name)
        for task_id in ids:
            state = states.get(task_id)
            ok = state == _OK_STATE
            error = None
            if not ok:
                # The traceback is in that element's log and nowhere else, so
                # the message is useless without the path to it.
                seen = state or "no state reported by sacct"
                if state and state not in _FAILED_STATES:
                    # Terminal, but not one of the states SLURM documents as a
                    # failure. Say so rather than assert a crash that may not
                    # have happened; the log is still the place to look.
                    seen = f"in the unrecognised terminal state {state}"
                error = (
                    f"array element {job_id}_{task_id} ended {seen}; "
                    f"the traceback is in {self.log_path(pass_name, task_id)}"
                )
            report.statuses.append(TaskStatus(task_id=task_id, ok=ok, error=error))
        return report

    # -- helpers ------------------------------------------------------------ #
    def _check_output(self, argv: Sequence[str], role: str) -> str:
        exe = str(argv[0])
        if shutil.which(exe) is None and not Path(exe).exists():
            raise RuntimeError(
                f"{exe!r} was not found on PATH, so the {role} step cannot "
                f"run. SlurmRunner has to be driven from a machine with a "
                f"SLURM client, which normally means a login or submit node. "
                f"Run it there, load the scheduler module, or set "
                f"runner: local in the config. To write the sbatch scripts "
                f"without submitting anything, use SlurmRunner(dry_run=True)."
            )
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{role} failed with exit code {proc.returncode}\n"
                f"  command: {' '.join(str(a) for a in argv)}\n"
                f"  stderr:  {proc.stderr.strip()}"
            )
        return proc.stdout

    def _template_dir(self) -> Path:
        if self.template_dir is not None:
            return self.template_dir
        # The templates live beside the package rather than inside it: they are
        # operator files, edited per cluster for module loads and constraints,
        # and a file someone is expected to edit should not be buried in an
        # installed package.
        here = Path(__file__).resolve()
        candidates = (here.parents[2] / "slurm", here.parents[1] / "slurm")
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"no sbatch templates found beside the package (looked in "
            f"{candidates[0]} and {candidates[1]}). Pass template_dir= to "
            f"point SlurmRunner at them."
        )

    def _require_cfg(self) -> RunConfig:
        if self.cfg is None:
            raise RuntimeError(
                "SlurmRunner has no RunConfig, so it cannot tell which "
                "partition, account or resources a pass needs. Construct it "
                "as SlurmRunner(cfg), or pass cfg= to bind()."
            )
        return self.cfg

    def _require_cfg_path(self) -> Path:
        if self.cfg_path is None:
            raise RuntimeError(
                "SlurmRunner has no config path, so an array element would "
                "have no way to rebuild its task. Call "
                "bind(cfg_path, level, iteration) before each pass, with a "
                "path the batch nodes can read."
            )
        return self.cfg_path
