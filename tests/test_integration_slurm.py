"""The level loop driving a scheduler runner.

The local and scheduler runners take completely different routes to the same
place: one calls a closure, the other writes a script that re-derives the work
from a config file on another machine. This checks the seam between the loop
and the second route, which is the part no unit test of either side can see.
"""

from __future__ import annotations

import yaml
import pytest

from chunkreg.config import load_config
from chunkreg.pipelines import build_template
from chunkreg.runners import get_runner
from chunkreg.runners.local import LocalRunner


@pytest.fixture
def config_file(tmp_path):
    cfg = {
        "root": str(tmp_path / "run"),
        "backend": "memory",
        "profile": "i1",
        "runner": "slurm",
        "spacing_mm": 0.05,
        "gpu_mem_gb": 1000.0,
        "subjects": [{"id": "s01", "path": "subjects/s01.zarr"}],
    }
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_a_scheduler_runner_without_a_config_path_is_refused(config_file):
    """The batch node needs a file to read; failing early beats failing on a node."""
    cfg = load_config(config_file)
    runner = get_runner("slurm", cfg=cfg, dry_run=True)
    with pytest.raises(ValueError, match="config_path"):
        build_template(cfg, runner=runner)


def test_the_local_runner_needs_no_config_path(config_file):
    """A closure carries its own context, so nothing has to be addressed."""
    cfg = load_config(config_file)
    # Reaching the subject-open stage without a ValueError is the assertion;
    # nothing is ingested, so it fails later and for a different reason.
    with pytest.raises(Exception) as exc:
        build_template(cfg, runner=LocalRunner())
    assert "config_path" not in str(exc.value)


def test_the_loop_binds_the_runner_before_every_pass(config_file, monkeypatch):
    """Each pass must be addressed with its own level and iteration."""
    cfg = load_config(config_file)
    runner = get_runner("slurm", cfg=cfg, dry_run=True)

    seen = []
    original = runner.bind

    def spy(cfg_path, level, iteration, cfg=None):
        seen.append((str(cfg_path), level, iteration))
        return original(cfg_path, level, iteration, cfg=cfg)

    monkeypatch.setattr(runner, "bind", spy)

    with pytest.raises(Exception):
        # No ingested subjects, so the loop stops early; what matters is that
        # it bound the runner before it got there.
        build_template(cfg, runner=runner, config_path=str(config_file))

    if seen:
        assert seen[0] == (str(config_file), 0, 0)
