#!/usr/bin/env python3
"""Supplemental read-only replay of frozen Methodology-v2 checkpoints.

No optimizer step, gradient, training, checkpoint selection, or sealed-file
write is reachable from this script. Outputs are confined to the existing
100pct_final_analysis/posthoc_metrics directory.
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.methodology_v2.experiment_executor import RepStore, ids_by_dataset
from src.methodology_v2.encoder import PCSTE, collate_representations
from src.methodology_v2.encoder.patchify import PATCH_F, PATCH_T, patchify
from src.methodology_v2.encoder.ssl_design import ReconstructionProbe
from src.methodology_v2.experiment.heads import CLASS_ORDERS, DatasetHeads, LABEL_FIELD
from src.methodology_v2.experiment.trainers import GRIDS, build_patch_mask
from src.methodology_v2.integrity import sha256_file

ORIGINAL = ROOT / "results/methodology_v2"
OUT = ROOT / "results/100pct_final_analysis/posthoc_metrics"
SCORES = OUT / "classification_scores"
RECON = OUT / "reconstruction"
MANIFESTS = ROOT / "methodology_v2/part3_windows"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
PRIMARY = ("JNU", "HIT", "MAFAULDA")


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def corr(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean(); yc = y - y.mean()
    den = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return np.nan if den == 0 else float((xc @ yc) / den)


def replay_classification(device):
    SCORES.mkdir(parents=True, exist_ok=True)
    audit = []
    for fold in (1, 2, 3):
        man = pd.read_csv(MANIFESTS / f"window_manifest_fold_{fold}.csv").set_index("window_id")
        test_ids = list(man.index[man.split == "test"])
        by_ds = ids_by_dataset(man, test_ids)
        store = RepStore(fold, man); store.preload(test_ids)
        for seed in (42, 1337, 2026):
            for arm in ("s0", "s1"):
                rid = f"{arm}_f{fold}_s{seed}_l100"
                source_dir = ORIGINAL / "downstream" / rid
                checkpoint = source_dir / "best.pt"
                saved = pd.read_csv(source_dir / "test_predictions.csv", dtype={"y_true": str, "y_pred": str}).set_index("window_id")
                ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert ck["run_id"] == rid
                encoder = PCSTE(); heads = DatasetHeads()
                encoder.load_state_dict(ck["encoder"]); heads.load_state_dict(ck["heads"])
                encoder.to(device).eval(); heads.to(device).eval()
                rows = []
                with torch.inference_mode():
                    for ds in DATASETS:
                        classes = CLASS_ORDERS[ds]
                        for lo in range(0, len(by_ds[ds]), 64):
                            chunk = by_ds[ds][lo:lo+64]
                            batch = collate_representations([store.rep(w) for w in chunk])
                            batch = {k:v.to(device) for k,v in batch.items()}
                            logits = heads(encoder(**batch)["global_embedding"], ds)
                            probs = torch.softmax(logits, dim=-1)
                            logits_np = logits.cpu().numpy().astype(np.float64)
                            probs_np = probs.cpu().numpy().astype(np.float64)
                            for i,wid in enumerate(chunk):
                                pred = classes[int(np.argmax(probs_np[i]))]
                                row = {"dataset":ds,"arm":arm.upper(),"fold":fold,"seed":seed,
                                       "sample_identifier":wid,"true_class":str(man.loc[wid, LABEL_FIELD[ds]]),
                                       "predicted_class":pred,"class_order_json":json.dumps(classes)}
                                for j,c in enumerate(classes):
                                    row[f"logit__{c}"] = logits_np[i,j]
                                    row[f"prob__{c}"] = probs_np[i,j]
                                rows.append(row)
                df = pd.DataFrame(rows)
                outpath = SCORES / f"{rid}_test_scores.csv.gz"
                df.to_csv(outpath, index=False, compression="gzip")
                q = df.set_index("sample_identifier")
                common = q.index.intersection(saved.index)
                mismatch = int((q.loc[common,"predicted_class"].astype(str) != saved.loc[common,"y_pred"].astype(str)).sum())
                sums=[]; finite=True
                for ds in DATASETS:
                    z=q[q.dataset==ds]
                    cols=[f"prob__{c}" for c in CLASS_ORDERS[ds]]
                    a=z[cols].to_numpy(float); sums.extend(a.sum(1)); finite &= bool(np.isfinite(a).all())
                rec={"run_id":rid,"arm":arm.upper(),"fold":fold,"seed":seed,
                     "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha256_file(checkpoint),
                     "manifest":str((MANIFESTS/f'window_manifest_fold_{fold}.csv').resolve()),
                     "n":len(df),"original_n":len(saved),"id_set_equal":set(q.index)==set(saved.index),
                     "hard_prediction_mismatches":mismatch,"probability_finite":finite,
                     "max_softmax_sum_error":float(np.max(np.abs(np.asarray(sums)-1))),
                     "status":"PASS" if mismatch==0 and len(df)==len(saved) and finite else "FAIL"}
                audit.append(rec); log(f"classification {rid}: {rec['status']} mismatch={mismatch}")
                del encoder, heads, ck; torch.cuda.empty_cache()
                if rec["status"] != "PASS":
                    pd.DataFrame(audit).to_csv(OUT/"classification_replay_audit.csv",index=False)
                    raise AssertionError(f"classification replay failed: {rid}")
    pd.DataFrame(audit).to_csv(OUT/"classification_replay_audit.csv",index=False)


def replay_reconstruction(device):
    RECON.mkdir(parents=True, exist_ok=True)
    detailed=[]; audit=[]
    for fold in (1,2,3):
        manpath=MANIFESTS/f"window_manifest_fold_{fold}.csv"
        man=pd.read_csv(manpath).set_index("window_id")
        val_ids=list(man.index[man.split=="validation"])
        by_ds=ids_by_dataset(man,val_ids)
        store=RepStore(fold,man); store.preload(val_ids)
        for seed in (42,1337,2026):
            rid=f"ssl_f{fold}_s{seed}"; cp=ORIGINAL/"ssl"/rid/"best.pt"
            completion=json.loads((ORIGINAL/"ssl"/rid/"completion.json").read_text())
            ck=torch.load(cp,map_location="cpu",weights_only=False); assert ck["run_id"]==rid
            model=ReconstructionProbe(PCSTE()); model.load_state_dict(ck["model"])
            model.to(device).eval()
            run_rows=[]
            with torch.inference_mode():
                for ds in DATASETS:
                    pick=by_ds[ds][len(by_ds[ds])//2]
                    picked=False
                    for lo in range(0,len(by_ds[ds]),32):
                        chunk=by_ds[ds][lo:lo+32]
                        reps=[store.rep(w) for w in chunk]
                        batch=collate_representations(reps)
                        fb=math.ceil(batch["spec"].shape[1]/PATCH_F); tp=math.ceil(batch["spec"].shape[2]/PATCH_T)
                        pm=build_patch_mask([(ds,w) for w in chunk],GRIDS,seed,0,True)[:,:fb,:tp]
                        batch_gpu={k:v.to(device) for k,v in batch.items()}; pm_gpu=pm.to(device)
                        out=model(**batch_gpu,patch_mask=pm_gpu)
                        pred=out["pred"].cpu().numpy(); lm=out["loss_mask"].cpu().numpy()
                        patches=patchify(batch["spec"],batch["cell_mask"])[0].squeeze(3).reshape(pred.shape).numpy()
                        for i,wid in enumerate(chunk):
                            mask=lm[i]; x=patches[i][mask].astype(np.float64); y=pred[i][mask].astype(np.float64)
                            err=x-y; sse=float(err@err); energy=float(x@x); xc=x-x.mean(); sst=float(xc@xc)
                            rx=rankdata(x,method="average"); ry=rankdata(y,method="average")
                            pear=corr(x,y); spear=corr(rx,ry)
                            row={"run_id":rid,"fold":fold,"seed":seed,"dataset":ds,"window_id":wid,
                                 "n_masked_valid_cells":len(x),"mse":float(np.mean(err*err)),
                                 "nmse":(sse/energy if energy>0 else np.nan),"mae":float(np.mean(np.abs(err))),
                                 "r2":(1-sse/sst if sst>0 else np.nan),"pearson_r":pear,"spearman_rho":spear,
                                 "target_energy_zero":energy==0,"target_constant":sst==0,
                                 "pearson_undefined":not np.isfinite(pear),"spearman_undefined":not np.isfinite(spear)}
                            run_rows.append(row); detailed.append(row)
                            if wid==pick and not picked:
                                np.savez_compressed(RECON/f"representative_{ds.lower()}_f{fold}_s{seed}.npz",
                                    window_id=np.array(wid),target_patches=patches[i],prediction_patches=pred[i],
                                    loss_mask=mask,patch_mask=pm[i].numpy(),original_spec=batch["spec"][i].numpy(),
                                    cell_mask=batch["cell_mask"][i].numpy())
                                picked=True
                        del out,batch_gpu,pm_gpu
            rdf=pd.DataFrame(run_rows); rdf.to_csv(RECON/f"{rid}_per_window_metrics.csv.gz",index=False,compression="gzip")
            recomputed={ds:float(rdf[rdf.dataset==ds].mse.mean()) for ds in DATASETS}
            maxdiff=max(abs(recomputed[ds]-completion["best_val_per_dataset_mse"][ds]) for ds in DATASETS)
            rec={"run_id":rid,"fold":fold,"seed":seed,"checkpoint":str(cp.resolve()),
                 "checkpoint_sha256":sha256_file(cp),"manifest":str(manpath.resolve()),
                 "mask_protocol":"validation_mask_seed(seed), epoch=0, M1_random, ratio=0.60",
                 "max_abs_mse_difference":maxdiff,"tolerance":1e-6,
                 "zero_energy_windows":int(rdf.target_energy_zero.sum()),
                 "constant_target_windows":int(rdf.target_constant.sum()),
                 "undefined_pearson_windows":int(rdf.pearson_undefined.sum()),
                 "undefined_spearman_windows":int(rdf.spearman_undefined.sum()),
                 "status":"PASS" if maxdiff<=1e-6 else "FAIL"}
            audit.append(rec); log(f"reconstruction {rid}: {rec['status']} max_mse_delta={maxdiff:.3g}")
            pd.DataFrame(detailed).to_csv(RECON/"reconstruction_per_window_all.csv.gz",index=False,compression="gzip")
            pd.DataFrame(audit).to_csv(OUT/"reconstruction_replay_audit.csv",index=False)
            del model,ck; torch.cuda.empty_cache()
            if rec["status"]!="PASS": raise AssertionError(f"reconstruction replay failed: {rid}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["classification","reconstruction","all"])
    ap.add_argument("--device",default="cuda"); a=ap.parse_args()
    if a.device.startswith("cuda"): assert torch.cuda.is_available()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"POSTHOC_SUPPLEMENTAL_EVALUATION").write_text(
        "Read-only inference from previously selected frozen checkpoints; no training or model selection.\n")
    if a.stage in ("classification","all"): replay_classification(a.device)
    if a.stage in ("reconstruction","all"): replay_reconstruction(a.device)

if __name__=="__main__": main()
