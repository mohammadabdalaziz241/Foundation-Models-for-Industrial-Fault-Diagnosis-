import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.multisource_paderborn_extension_v1.cwru_ensemble import hard_vote,ensemble_predict
from src.joint_ssl import mixed_balanced_batch_indices

def test_hard_vote_validation_tiebreak_is_deterministic():
 p=[np.array([0,1]),np.array([1,1])]
 assert hard_vote(p,[.8,.9]).tolist()==[1,1]

def test_probability_and_logit_ensemble_shapes():
 z=[np.array([[2.,0.,0.,0.],[0.,2.,0.,0.]]),np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])]
 for method in ("soft","logit","weighted"):
  assert ensemble_predict(method,z,[.7,.6]).tolist()==[0,1]

def test_two_source_balanced_batches_retain_effective_batch_64():
 rng=np.random.default_rng(42)
 batches=list(mixed_balanced_batch_indices([101,1001],64,rng,n_batches=7))
 assert len(batches)==7 and all(sum(len(x) for x in b)==64 for b in batches)
 assert all([len(x) for x in b]==[32,32] for b in batches)


def test_extension_holm_and_all_tie_statistics():
    import importlib.util
    path = ROOT / "experiments/multisource_paderborn_extension_v1/analysis.py"
    spec = importlib.util.spec_from_file_location("multisource_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"raw_p": p} for p in (0.04, 0.001, 0.02)]
    module.holm(rows)
    ordered = sorted(rows, key=lambda row: row["raw_p"])
    assert all(0 <= row["holm_p"] <= 1 for row in rows)
    assert [row["holm_p"] for row in ordered] == sorted(row["holm_p"] for row in ordered)
    result = module.paired([0.5] * 12, [0.5] * 12, "tie", "test")
    assert result["raw_p"] == 1.0
    assert result["ties"] == 12


def test_verifier_uses_repository_canonical_test_bearings():
    text = (ROOT / "experiments/multisource_paderborn_extension_v1/analysis.py").read_text()
    assert "from src.cv_paderborn import TEST_BEARINGS" in text
    assert 'set(entry["test_bearings"])!=set(TEST_BEARINGS)' in text
