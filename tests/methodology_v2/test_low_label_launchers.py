"""Static publication tests for recovered low-label launchers.

These tests parse source only; they do not train models or evaluate TEST data.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXPECTED = {
    "run_methodology_v2_1pct_extension.py": (0.01, "frac_1", "l001"),
    "run_methodology_v2_5pct.py": (0.05, "frac_5", "l005"),
    "run_methodology_v2_10pct.py": (0.10, "frac_10", "l010"),
}


def assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment: {name}")


@pytest.mark.parametrize("filename,expected", EXPECTED.items())
def test_low_label_launcher_contract(filename, expected):
    path = REPO / "scripts" / filename
    source = path.read_text()
    tree = ast.parse(source)
    fraction, column, suffix = expected
    assert assignment(tree, "NOMINAL_FRACTION") == fraction
    assert assignment(tree, "FRACTION_COLUMN") == column
    assert f"_{suffix}" in source
    assert "PCSTE_RESULTS_ROOT" in source
    assert "scripts/methodology_v2/experiment_executor.py" in source
    assert "SupervisedTrainer" in source
    assert "SupervisedSampler" in source
    assert "DOWNSTREAM_EPOCHS" in source
    assert "for fold in FOLDS" in source
    assert "for seed in SEEDS" in source


def test_frozen_shared_constants():
    labels = ast.parse((REPO / "src/methodology_v2/experiment/label_subsets.py").read_text())
    trainers = ast.parse((REPO / "src/methodology_v2/experiment/trainers.py").read_text())
    assert assignment(labels, "FOLDS") == (1, 2, 3)
    assert assignment(labels, "SEEDS") == (42, 1337, 2026)
    assert assignment(trainers, "DOWNSTREAM_EPOCHS") == 50


def test_shared_executor_imports_without_execution():
    path = REPO / "scripts/methodology_v2/experiment_executor.py"
    tree = ast.parse(path.read_text())
    assert assignment(tree, "DATASETS") == ("CWRU", "JNU", "HIT", "MAFAULDA")
    assert assignment(tree, "MICRO_BATCH") == 32
