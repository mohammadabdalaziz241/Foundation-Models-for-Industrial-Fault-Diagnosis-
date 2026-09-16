"""Non-training tests for blocked XJTU architecture-comparison preparation."""
import json,subprocess,sys
from pathlib import Path
import numpy as np,pytest,torch
ROOT=Path(__file__).resolve().parents[1]; HERE=ROOT/"experiments/xjtu_architecture_comparison_v1"; sys.path[:0]=[str(ROOT),str(HERE)]
from protocol import ComputeBudget,assert_split,assert_temporal_isolation,candidate_splits,generate_labels,load_source_index,validate_target
from src.models import build_model
from src.models.masked_ssl import MaskedSSL
from src.models.tcn1d import receptive_field

def test_target_is_explicitly_blocked_and_labels_refuse():
 t=json.loads((HERE/"target_definition.json").read_text()); assert validate_target(t)
 with pytest.raises(RuntimeError,match="blocked target definition"): generate_labels({"bearings":np.array(["Bearing1_1"])},t)
 assert not list((ROOT/"data/processed_xjtu_sy").glob("**/labels.npy"))

def test_candidate_folds_are_bearing_and_temporally_safe():
 rows=load_source_index(ROOT/"data/manifests/xjtu_sy_manifest.csv")
 folds=candidate_splits(); assert len(folds)==5
 for f in folds: assert_split(f); assert_temporal_isolation(f,rows)
 assert {b for f in folds for b in f["test_bearings"]}=={f"Bearing{c}_{i}" for c in (1,2,3) for i in range(1,6)}

def test_equal_compute_and_h_remainder_accounting():
 b=ComputeBudget(); assert b.total_steps==13860; assert b.examples_per_epoch==29568; assert b.effective_examples_seen==887040
 base,rem=divmod(64,3); assert (base,rem)==(21,1)
 totals=[0,0,0]
 for i in range(462):
  counts=[base]*3; counts[i%3]+=rem; assert sum(counts)==64 and max(counts)-min(counts)<=1
  totals=[a+x for a,x in zip(totals,counts)]
 assert totals==[9856,9856,9856]

def test_architecture_ssl_and_freeze_contracts_without_optimisation():
 x=torch.randn(2,1,1024)
 for a in ["lstm","tcn","tcn_rf1021","cnn1d","inceptiontime"]:
  m=build_model(a,num_classes=3).eval(); before={n:p.detach().clone() for n,p in m.encoder.named_parameters()}
  with torch.no_grad(): assert m(x).shape==(2,3); assert m.encode(x).shape==(2,128); recon,loss=MaskedSSL(encoder_model=m)(x)
  assert recon.shape==x.shape and torch.isfinite(loss)
  for p in m.parameters(): p.requires_grad=False
  for p in m.head.parameters(): p.requires_grad=True
  with torch.no_grad(): m(x)
  assert all(torch.equal(before[n],p) for n,p in m.encoder.named_parameters())
 assert receptive_field(3,(1,2,4,8))==61 and receptive_field(3,(1,2,4,8,16,32,64,128))==1021

def test_manifest_and_launcher_block_execution():
 manage=HERE/"manage.py"; subprocess.run([sys.executable,str(manage),"manifest"],cwd=ROOT,check=True,capture_output=True,text=True)
 m=json.loads((HERE/"manifest.json").read_text()); assert len(m["entries"])==600; assert {e["status"] for e in m["entries"]}=={"blocked_target_definition"}
 dry=subprocess.run([sys.executable,str(manage),"launch","--dry-run"],cwd=ROOT,check=True,capture_output=True,text=True); assert "0 executable tasks" in dry.stdout and "DRY RUN ONLY" in dry.stdout
 execute=subprocess.run([sys.executable,str(manage),"launch","--execute"],cwd=ROOT,capture_output=True,text=True); assert execute.returncode!=0 and "REFUSING EXECUTION" in execute.stderr
