"""No module may reference a name that is not bound.

A misspelled or renamed variable is a ``NameError`` raised only when control
reaches that line. The sweep runners are the worst case for that: a branch taken
only by one config set can sit broken through a whole grid and surface as a
method that failed 18 seconds into a run, after the index was built and the
weights were loaded.

The same fault is visible statically in under a second, so it is checked here
rather than paid for in GPU time.

Deferred-evaluation closures are the one thing that reads as a false positive:
a ``lambda`` closing over a name that is ``del``-ed later in the same scope is
flagged even though the call happens before the deletion. The runners bind those
through default arguments (``lambda q, p=pipeline: ...``), which both silences
the check and captures the value at definition time.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PACKAGES = [
    "analysis",
    "core",
    "dp_methods",
    "experiments",
    "metrics",
    "scripts",
    "tests",
    "vanilla",
]


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
def test_no_undefined_names():
    repo_root = REPO_ROOT
    targets = [str(repo_root / name) for name in PACKAGES if (repo_root / name).exists()]
    result = subprocess.run(
        ["ruff", "check", "--select", "F821", "--output-format", "concise", *targets],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, f"undefined names:\n{result.stdout}{result.stderr}"
