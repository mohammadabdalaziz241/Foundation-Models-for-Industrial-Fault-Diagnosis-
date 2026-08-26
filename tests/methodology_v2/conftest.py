"""Shared test helpers for methodology_v2.

assert_imports_clean: verifies in an ISOLATED subprocess that importing
the given audit modules pulls in no forbidden module (torch, legacy
pipelines). Subprocess isolation is required because Part-5B encoder
tests legitimately import torch in the same pytest session; the audit
guarantee is about what the audit modules THEMSELVES import, not about
the shared interpreter state.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def assert_imports_clean(module_names: list[str]) -> None:
    code = (
        "import sys; sys.path.insert(0, {root!r})\n"
        "for m in {mods!r}:\n"
        "    __import__(m)\n"
        "from src.methodology_v2 import FORBIDDEN_IMPORTS\n"
        "bad = [m for m in FORBIDDEN_IMPORTS if m in sys.modules]\n"
        "assert not bad, f'forbidden modules loaded: {{bad}}'\n"
    ).format(root=str(REPO_ROOT), mods=list(module_names))
    res = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert res.returncode == 0, (
        f"audit-import isolation check failed:\n{res.stderr[-2000:]}")
