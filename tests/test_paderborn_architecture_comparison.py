"""Non-training validation for PAC-v1 (no optimiser and no real-data forward loop)."""
import json, subprocess, sys, tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.cv_paderborn import TEST_BEARINGS, make_folds, split_ssl_val_bearings
from src.models import build_model
from src.models.masked_ssl import MaskedSSL
from src.models.tcn1d import receptive_field
from experiments.paderborn_architecture_comparison_v1.analysis import classification_complete
ARCHS=["lstm","tcn","tcn_rf1021","cnn1d","inceptiontime"]
def test_architecture_contracts_and_ssl_shapes():
 x=torch.randn(2,1,1024)
 for a in ARCHS:
  m=build_model(a,num_classes=3).eval()
  with torch.no_grad():
   assert m(x).shape==(2,3); assert m.encode(x).shape==(2,128); assert m.forward_sequence(x).ndim==3
   ssl=MaskedSSL(encoder_model=m,patch_len=32,mask_ratio=.5); recon,loss=ssl(x)
  assert recon.shape==x.shape and torch.isfinite(loss)
  before={n:p.detach().clone() for n,p in m.encoder.named_parameters()}
  for p in m.parameters(): p.requires_grad=False
  for p in m.head.parameters(): p.requires_grad=True
  with torch.no_grad(): m(x)
  assert all(torch.equal(before[n],p) for n,p in m.encoder.named_parameters())
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/"e.pt"; ssl.export_encoder_weights(path); m2=build_model(a,num_classes=3); m2.encoder.load_state_dict(torch.load(path,weights_only=True),strict=True)
def test_folds_ssl_leakage_and_rf():
 assert receptive_field(3,(1,2,4,8))==61
 assert receptive_field(3,(1,2,4,8,16,32,64,128))==1021
 for rep in make_folds(4,3,42):
  vals=[]
  for f in rep:
   tr=set(f["train_bearings"]); va=set(f["val_bearings"]); assert not tr&va; assert not (tr|va)&set(TEST_BEARINGS); vals+=list(va)
   st,sv=split_ssl_val_bearings(f["train_bearings"],.2,123); assert not set(st)&set(sv); assert set(st)|set(sv)==tr; assert not (set(st)|set(sv))&va
  assert len(vals)==len(set(vals))==21
def test_manifest_and_default_launcher_are_safe():
 manage=ROOT/"experiments/paderborn_architecture_comparison_v1/manage.py"
 subprocess.run([sys.executable,str(manage),"manifest"],cwd=ROOT,check=True,capture_output=True,text=True)
 m=json.loads((manage.parent/"manifest.json").read_text()); assert len(m["entries"])==240
 assert sum(e["status"]=="reusable" for e in m["entries"])==36
 assert sum(e["status"]=="prepared" for e in m["entries"])==204
 out=subprocess.run([sys.executable,str(manage),"launch"],cwd=ROOT,check=True,capture_output=True,text=True).stdout
 assert "DRY RUN ONLY" in out and "204 prepared tasks" in out

def test_completion_inference_handles_ssl_and_statusless_classification():
 ssl={"status":"complete","regime":"ssl_pretrain"}
 classification={"repeat":0,"fold":0,"train_seed":42,"selected_epoch":3,
  "final_bb_f1":.5,"final_window_f1":.49,"final_accuracy":.6,
  "final_per_class_f1":{"Normal":.7,"Inner Race":.4,"Outer Race":.37},
  "val_bearings":["K002"],"train_time_s":12.0}
 assert ssl["status"] == "complete"
 assert "status" not in classification and classification_complete(classification)
 assert not classification_complete({**classification,"final_bb_f1":float("nan")})
