#!/usr/bin/env python3
"""Fail-closed completeness gate and aggregation for corrected InceptionTime."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/baselines/inceptiontime_four_domain"
OUT = ROOT / "aggregate"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
CELLS = tuple((f, s) for f in (1, 2, 3) for s in (42, 1337, 2026))

def finite(x):
    x = float(x)
    if not np.isfinite(x): raise ValueError(f"non-finite metric: {x}")
    return x

def read_cell(fold, seed):
    run = f"inceptiontime_f{fold}_s{seed}_l100"
    d = ROOT / run
    missing = [x for x in ("best.pt", "test_predictions.csv", "test_report.json") if not (d/x).is_file()]
    if missing: raise FileNotFoundError(f"{run}: missing {missing}")
    report = json.loads((d/"test_report.json").read_text())
    per = report.get("per_dataset", {})
    if set(per) != set(DATASETS): raise ValueError(f"{run}: datasets={sorted(per)}")
    row = {"run_id": run, "fold": fold, "seed": seed}
    f1s, aucs = [], []
    for ds in DATASETS:
        f1 = finite(per[ds]["macro_f1"]); f1s.append(f1); row[f"{ds}_macro_f1"] = f1
        auc = finite(per[ds]["macro_roc_auc_ovr"]); aucs.append(auc); row[f"{ds}_macro_auc"] = auc
    row["macro4_f1"] = float(np.mean(f1s)); row["macro4_auc"] = float(np.mean(aucs))
    if not np.isclose(finite(report["macro4_f1"]), row["macro4_f1"], rtol=0, atol=1e-12):
        raise ValueError(f"{run}: stored macro4_f1 mismatch")
    if not np.isclose(finite(report["macro4_auc"]), row["macro4_auc"], rtol=0, atol=1e-12):
        raise ValueError(f"{run}: stored macro4_auc mismatch")
    return row

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--status", action="store_true"); args = ap.parse_args()
    if args.status:
        for f,s in CELLS:
            d=ROOT/f"inceptiontime_f{f}_s{s}_l100"; state=d/"state.json"
            status=json.loads(state.read_text()).get("status","UNKNOWN") if state.is_file() else "NOT_STARTED"
            epoch=sum(1 for _ in open(d/"epoch_metrics.jsonl")) if (d/"epoch_metrics.jsonl").is_file() else 0
            print(f"f{f} s{s}: {status:28s} epoch={epoch:2d}/50 host={json.loads(state.read_text()).get('host','-') if state.is_file() else '-'}")
        return
    rows=[read_cell(f,s) for f,s in CELLS]
    df=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); df.to_csv(OUT/"per_cell.csv",index=False)
    metrics=[f"{ds}_macro_f1" for ds in DATASETS]+["macro4_f1"]+[f"{ds}_macro_auc" for ds in DATASETS]+["macro4_auc"]
    stats={m:{"mean":float(df[m].mean()),"sample_sd":float(df[m].std(ddof=1)),"n":9} for m in metrics}
    pd.DataFrame([{"metric":m,**stats[m]} for m in metrics]).to_csv(OUT/"aggregate.csv",index=False)
    (OUT/"aggregate.json").write_text(json.dumps({"n_cells":9,"datasets":DATASETS,"statistics":stats},indent=2)+"\n")
    lines=["Corrected four-domain InceptionTime (9 fold×seed cells)",""]+[f"{m}: {stats[m]['mean']:.6f} ± {stats[m]['sample_sd']:.6f}" for m in metrics]
    (OUT/"SUMMARY.txt").write_text("\n".join(lines)+"\n"); print("\n".join(lines))
if __name__ == "__main__": main()
