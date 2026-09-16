import csv,json
from pathlib import Path
import numpy as np,pytest,torch
from scripts.run_foundation_experiment import supervised_loss,validate_combo
from src.foundation_dataset import DatasetBearingBalancedBatchSampler
from src.foundation_heads import FoundationCNN1D
from src.foundation_metrics import dataset_metrics,metrics_from_rows,replay,write_group_metric_tables

def test_combinations():
 validate_combo("S0","scratch");validate_combo("S1","ssl");validate_combo("S1","linear");validate_combo("S2","full")
 for pair in (("S0","linear"),("S1","scratch"),("S2","scratch")):
  with pytest.raises(ValueError):validate_combo(*pair)

def test_routing_balanced_loss_and_freezing():
 model=FoundationCNN1D();x=torch.randn(4,1,1024);datasets=["cwru","cwru","paderborn","paderborn"];y=torch.tensor([0,1,0,1])
 routed=model(x,datasets);assert {k:len(v[0]) for k,v in routed.items()}=={"cwru":2,"paderborn":2}
 loss,parts=supervised_loss(model,x,y,datasets,{"cwru":torch.ones(4),"paderborn":torch.ones(3)})
 assert torch.allclose(loss,torch.stack(list(parts.values())).mean())
 before={k:v.clone() for k,v in model.encoder.state_dict().items()};model.freeze_encoder(True)
 opt=torch.optim.Adam(model.heads.parameters());opt.zero_grad();loss,_=supervised_loss(model,x,y,datasets,{"cwru":torch.ones(4),"paderborn":torch.ones(3)});loss.backward();opt.step()
 assert all(torch.equal(v,model.encoder.state_dict()[k]) for k,v in before.items())
 assert all(p.grad is None for p in model.encoder.parameters())

def test_sampler_and_metrics_replay(tmp_path):
 rows=[{"dataset":d,"bearing_id":b} for d in ("cwru","paderborn","xjtu_sy") for b in ("a","b") for _ in range(3)]
 sampler=DatasetBearingBalancedBatchSampler(rows,6,2,42);list(sampler);tot={d:sum(v for (dd,b),v in sampler.exposure.items() if dd==d) for d in ("cwru","paderborn","xjtu_sy")}
 assert len(set(tot.values()))==1
 pred=tmp_path/"p.csv";fields=["dataset","true_index","predicted_index"]
 with pred.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{"dataset":"cwru","true_index":0,"predicted_index":0},{"dataset":"paderborn","true_index":0,"predicted_index":0}])
 metrics=metrics_from_rows(list(csv.DictReader(pred.open())));stored=tmp_path/"m.json";stored.write_text(json.dumps(metrics))
 assert replay(pred,stored)==metrics;assert metrics["combined"]["mean_dataset_macro_f1"]==(metrics["datasets"]["cwru"]["macro_f1"]+metrics["datasets"]["paderborn"]["macro_f1"])/2
 assert np.asarray(metrics["datasets"]["cwru"]["confusion_matrix_normalized"]).shape==(4,4)

def test_sampler_epoch_audits_are_balanced_and_change():
 rows=[{"dataset":d,"bearing_id":b,"recording_id":b} for d in ("cwru","paderborn") for b in ("a","b","c") for _ in range(4)]
 sampler=DatasetBearingBalancedBatchSampler(rows,12,3,42);sampler.set_epoch(1);first=list(sampler);first_exp=dict(sampler.exposure)
 sampler.set_epoch(2);second=list(sampler);second_exp=dict(sampler.exposure)
 assert first!=second and first_exp!=second_exp
 assert len({sum(v for (dd,_),v in sampler.exposure.items() if dd==d) for d in sampler.datasets})==1

def test_sampler_uses_cwru_recording_as_group():
 rows=[{"dataset":"cwru","bearing_id":"shared","recording_id":r} for r in ("r1","r2") for _ in range(3)]
 sampler=DatasetBearingBalancedBatchSampler(rows,20,10,42);list(sampler)
 assert {g for (d,g) in sampler.exposure}=={"r1","r2"}
