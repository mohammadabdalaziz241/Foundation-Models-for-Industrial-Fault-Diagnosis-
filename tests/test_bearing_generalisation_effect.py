"""Tests for the frozen primary-effect calculation (Stage 3.2).

These pin the arithmetic that decides success, so it cannot drift once real scores
exist. Every threshold and cell identifier is asserted against a literal, not against
the module's own constants, so a change to the frozen declaration fails loudly here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bearing_generalisation_effect import (
    DECLARED_FOLDS, DECLARED_SEEDS, MINIMUM_EFFECT, PRIMARY_DECLARATION,
    PrimaryEffectError, Score, bearing_generalisation_gap, compute_primary_effect,
    format_primary_effect, scores_from_records,
)


def _cell(fold, arm, strategy, seed, value, regime="R1", setting="setting_a"):
    return Score(fold=fold, arm=arm, strategy=strategy, regime=regime,
                 setting=setting, seed=seed, value=value)


def _grid(treatment_by_fold, control_by_fold, seeds=DECLARED_SEEDS):
    """Build a complete 4-fold x 3-seed score set from per-fold seed-mean targets."""
    out = []
    for fold in DECLARED_FOLDS:
        for seed in seeds:
            out.append(_cell(fold, "P2", "F3", seed, treatment_by_fold[fold]))
            out.append(_cell(fold, "P0", "scratch", seed, control_by_fold[fold]))
    return out


# ===========================================================================
# the frozen declaration
# ===========================================================================

def test_primary_declaration_matches_the_amendment_verbatim():
    assert PRIMARY_DECLARATION["treatment"] == "P2/F3"
    assert PRIMARY_DECLARATION["control"] == "P0/scratch"
    assert PRIMARY_DECLARATION["regime"] == "R1"
    assert PRIMARY_DECLARATION["setting"] == "setting_a"
    assert PRIMARY_DECLARATION["metric"] == "bb_f1"
    assert PRIMARY_DECLARATION["minimum_effect"] == 0.02
    assert PRIMARY_DECLARATION["require_positive_in_all_folds"] is True
    assert PRIMARY_DECLARATION["folds"] == ["fold0", "fold1", "fold2", "fold3"]
    assert PRIMARY_DECLARATION["seeds"] == [42, 43, 44]
    assert PRIMARY_DECLARATION["seeds_are_independent_units"] is False
    assert MINIMUM_EFFECT == 0.02


def test_no_significance_machinery_is_exposed():
    import src.bearing_generalisation_effect as mod
    names = dir(mod)
    for banned in ("wilcoxon", "ttest", "p_value", "pvalue"):
        assert not any(banned in n.lower() for n in names), banned


# ===========================================================================
# the arithmetic
# ===========================================================================

def test_delta_is_the_mean_of_paired_fold_differences():
    t = {"fold0": 0.60, "fold1": 0.55, "fold2": 0.70, "fold3": 0.65}
    c = {"fold0": 0.50, "fold1": 0.50, "fold2": 0.60, "fold3": 0.60}
    effect = compute_primary_effect(_grid(t, c))
    assert effect.fold_differences == pytest.approx(
        {"fold0": 0.10, "fold1": 0.05, "fold2": 0.10, "fold3": 0.05})
    assert effect.delta == pytest.approx(0.075)
    assert effect.n_independent_units == 4


def test_seeds_are_averaged_within_fold_not_pooled_as_units():
    scores = []
    for fold in DECLARED_FOLDS:
        for seed, v in zip(DECLARED_SEEDS, (0.50, 0.60, 0.70)):   # seed mean 0.60
            scores.append(_cell(fold, "P2", "F3", seed, v))
            scores.append(_cell(fold, "P0", "scratch", seed, 0.50))
    effect = compute_primary_effect(scores)
    assert effect.treatment_fold_means["fold0"] == pytest.approx(0.60)
    assert effect.n_independent_units == 4, "n must be folds, never folds x seeds"
    assert effect.seeds_per_cell == {f: 3 for f in DECLARED_FOLDS}


def test_success_requires_both_conditions():
    # Delta well above threshold, but one fold is negative -> failure
    t = {"fold0": 0.70, "fold1": 0.70, "fold2": 0.70, "fold3": 0.49}
    c = {f: 0.50 for f in DECLARED_FOLDS}
    effect = compute_primary_effect(_grid(t, c))
    assert effect.meets_minimum_effect is True
    assert effect.positive_in_all_folds is False
    assert effect.success is False

    # Positive everywhere, but Delta below threshold -> failure
    t = {f: 0.51 for f in DECLARED_FOLDS}
    effect = compute_primary_effect(_grid(t, c))
    assert effect.delta == pytest.approx(0.01)
    assert effect.positive_in_all_folds is True
    assert effect.meets_minimum_effect is False
    assert effect.success is False

    # Both satisfied -> success
    t = {f: 0.53 for f in DECLARED_FOLDS}
    effect = compute_primary_effect(_grid(t, c))
    assert effect.success is True


def test_delta_exactly_at_the_threshold_succeeds():
    c = {f: 0.50 for f in DECLARED_FOLDS}
    t = {f: 0.52 for f in DECLARED_FOLDS}
    effect = compute_primary_effect(_grid(t, c))
    assert effect.delta == pytest.approx(0.02)
    assert effect.meets_minimum_effect is True
    assert effect.success is True


def test_a_zero_fold_difference_is_not_positive():
    c = {f: 0.50 for f in DECLARED_FOLDS}
    t = {"fold0": 0.50, "fold1": 0.60, "fold2": 0.60, "fold3": 0.60}
    effect = compute_primary_effect(_grid(t, c))
    assert effect.fold_differences["fold0"] == pytest.approx(0.0)
    assert effect.positive_in_all_folds is False
    assert effect.success is False


# ===========================================================================
# refusals
# ===========================================================================

def test_refuses_a_missing_fold():
    t = {f: 0.60 for f in DECLARED_FOLDS}
    c = {f: 0.50 for f in DECLARED_FOLDS}
    scores = [s for s in _grid(t, c) if s.fold != "fold3"]
    with pytest.raises(PrimaryEffectError, match="no score for fold=fold3"):
        compute_primary_effect(scores)


def test_refuses_a_missing_seed():
    t = {f: 0.60 for f in DECLARED_FOLDS}
    c = {f: 0.50 for f in DECLARED_FOLDS}
    scores = [s for s in _grid(t, c) if not (s.fold == "fold1" and s.seed == 44)]
    with pytest.raises(PrimaryEffectError, match="seeds"):
        compute_primary_effect(scores)


def test_refuses_unpaired_treatment_and_control_seeds():
    scores = []
    for fold in DECLARED_FOLDS:
        for seed in DECLARED_SEEDS:
            scores.append(_cell(fold, "P2", "F3", seed, 0.6))
        for seed in (42, 43, 99):
            scores.append(_cell(fold, "P0", "scratch", seed, 0.5))
    with pytest.raises(PrimaryEffectError, match="seeds"):
        compute_primary_effect(scores, required_seeds=None)


def test_refuses_duplicate_seeds():
    t = {f: 0.60 for f in DECLARED_FOLDS}
    c = {f: 0.50 for f in DECLARED_FOLDS}
    scores = _grid(t, c) + [_cell("fold0", "P2", "F3", 42, 0.99)]
    with pytest.raises(PrimaryEffectError, match="duplicate seeds"):
        compute_primary_effect(scores)


def test_refuses_empty_input():
    with pytest.raises(PrimaryEffectError, match="no scores"):
        compute_primary_effect([])


def test_wrong_setting_is_not_silently_substituted():
    t = {f: 0.60 for f in DECLARED_FOLDS}
    c = {f: 0.50 for f in DECLARED_FOLDS}
    setting_b_only = [Score(s.fold, s.arm, s.strategy, s.regime, "setting_b", s.seed, s.value)
                      for s in _grid(t, c)]
    with pytest.raises(PrimaryEffectError, match="no score for"):
        compute_primary_effect(setting_b_only)


def test_wrong_regime_is_not_silently_substituted():
    t = {f: 0.60 for f in DECLARED_FOLDS}
    c = {f: 0.50 for f in DECLARED_FOLDS}
    r5_only = [Score(s.fold, s.arm, s.strategy, "R5", s.setting, s.seed, s.value)
               for s in _grid(t, c)]
    with pytest.raises(PrimaryEffectError, match="no score for"):
        compute_primary_effect(r5_only)


# ===========================================================================
# partial (smoke-test) mode
# ===========================================================================

def test_single_fold_single_seed_never_reports_primary_success():
    scores = [_cell("fold0", "P2", "F3", 42, 0.90),
              _cell("fold0", "P0", "scratch", 42, 0.10)]
    effect = compute_primary_effect(scores, folds=("fold0",), required_seeds=None,
                                    allow_partial_seeds=True)
    assert effect.delta == pytest.approx(0.80)
    assert effect.meets_minimum_effect is True
    assert effect.positive_in_all_folds is True
    assert effect.success is False, "an incomplete fold set can never be a primary result"
    assert effect.n_independent_units == 1


# ===========================================================================
# helpers
# ===========================================================================

def test_scores_from_records_round_trips():
    records = [{"fold": "fold0", "arm": "P2", "strategy": "F3", "regime": "R1",
                "setting": "setting_a", "seed": 42, "bb_f1": 0.42}]
    assert scores_from_records(records)[0] == _cell("fold0", "P2", "F3", 42, 0.42)


def test_bearing_generalisation_gap_is_a_minus_b():
    assert bearing_generalisation_gap(0.80, 0.30) == pytest.approx(0.50)
    assert bearing_generalisation_gap(0.30, 0.80) == pytest.approx(-0.50)


def test_format_mentions_no_significance():
    t = {f: 0.60 for f in DECLARED_FOLDS}
    c = {f: 0.50 for f in DECLARED_FOLDS}
    text = format_primary_effect(compute_primary_effect(_grid(t, c)))
    assert "No significance test" in text
    assert "p =" not in text and "p-value" not in text
