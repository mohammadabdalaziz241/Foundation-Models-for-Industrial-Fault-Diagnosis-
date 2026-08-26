#!/usr/bin/env python3
"""Read-only extraction of frozen Methodology-v2 100%-label results.

The script never imports or executes the training/evaluation executor, never
loads a checkpoint, and writes only beneath results/100pct_final_analysis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/methodology_v2"
REGISTRY = ROOT / "methodology_v2/part5_experiment_registry"
SPLITS = ROOT / "methodology_v2/part3_windows"
OUT = ROOT / "results/100pct_final_analysis"
TABLES = OUT / "tables"
CM_IND = OUT / "figures/confusion_matrices/individual"
CM_AGG = OUT / "figures/confusion_matrices/aggregate"
ROC_IND = OUT / "figures/roc_curves/individual"
ROC_AGG = OUT / "figures/roc_curves/aggregate"
RECON_FIG = OUT / "figures/reconstruction"
PROV = OUT / "provenance"

DATASETS = ["JNU", "HIT", "MAFAULDA"]
DISPLAY = {"JNU": "JNU", "HIT": "HIT", "MAFAULDA": "MaFaulDa"}
ARMS, FOLDS, SEEDS = ["S0", "S1"], [1, 2, 3], [42, 1337, 2026]
CLASSES = {
    "JNU": ["n", "ib", "ob", "tb"],
    "HIT": ["0", "1", "2"],
    "MAFAULDA": ["normal", "imbalance", "horizontal-misalignment",
                  "vertical-misalignment", "underhang/ball_fault",
                  "underhang/cage_fault", "underhang/outer_race",
                  "overhang/ball_fault", "overhang/cage_fault",
                  "overhang/outer_race"],
}
SEMANTIC = {
    "JNU": {"n": "healthy", "ib": "inner race fault",
            "ob": "outer race fault", "tb": "rolling element fault"},
    "HIT": {"0": "healthy", "1": "inner race fault",
            "2": "outer race fault"},
    "MAFAULDA": {c: c.replace("_", " ") for c in CLASSES["MAFAULDA"]},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_id(arm: str, fold: int, seed: int) -> str:
    return f"{arm.lower()}_f{fold}_s{seed}_l100"


def cm_metrics(cm: np.ndarray) -> dict:
    cm = cm.astype(float)
    tp = np.diag(cm)
    support = cm.sum(1)
    predicted = cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    n = support.sum()
    weights = np.divide(support, n, out=np.zeros_like(support), where=n > 0)
    return {
        "accuracy": float(tp.sum() / n),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "precision": precision, "recall": recall, "f1": f1,
        "support": support.astype(int), "n": int(n),
    }


def confusion(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        out[idx[str(t)], idx[str(p)]] += 1
    return out


def plot_cm(cm, labels, title, path, normalized=False):
    vals = cm.astype(float)
    if normalized:
        den = vals.sum(1, keepdims=True)
        vals = np.divide(vals, den, out=np.zeros_like(vals), where=den > 0)
    n = len(labels)
    fig_w = max(6.2, 0.72 * n + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.86))
    im = ax.imshow(vals, cmap="Blues", vmin=0,
                   vmax=(1 if normalized else max(1, vals.max())))
    ax.set_xticks(range(n), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    threshold = (0.5 if normalized else vals.max() / 2)
    for i in range(n):
        for j in range(n):
            text = f"{vals[i,j]:.2f}" if normalized else f"{int(vals[i,j])}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=(7 if n > 5 else 9),
                    color="white" if vals[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def fmt(ms):
    return f"{ms[0]:.4f} ± {ms[1]:.4f}"


def main():
    for d in [TABLES, CM_IND, CM_AGG, ROC_IND, ROC_AGG, RECON_FIG, PROV]:
        d.mkdir(parents=True, exist_ok=True)

    source_files, log, inventory, detailed, per_class, discrepancies = [], [], [], [], [], []
    registry = pd.read_csv(REGISTRY / "main_run_registry.csv")
    registry = registry[(registry.arm.isin(ARMS)) & (registry.label_fraction == 1.0)]
    expected_ids = {run_id(a, f, s) for a in ARMS for f in FOLDS for s in SEEDS}
    registered_ids = set(registry.run_id)
    log.append(f"Expected downstream runs: {len(expected_ids)}")
    log.append(f"Registered 100%-label S0/S1 runs: {len(registered_ids)}")
    if expected_ids != registered_ids:
        log.append(f"Registry missing: {sorted(expected_ids-registered_ids)}")
        log.append(f"Registry extras: {sorted(registered_ids-expected_ids)}")

    run_predictions = {}
    run_cms = {}
    split_cache = {}
    for arm in ARMS:
        for fold in FOLDS:
            man_path = SPLITS / f"window_manifest_fold_{fold}.csv"
            if fold not in split_cache:
                split_cache[fold] = pd.read_csv(man_path)
                source_files.append(man_path)
            test_ids = set(split_cache[fold].loc[split_cache[fold].split == "test", "window_id"])
            for seed in SEEDS:
                rid = run_id(arm, fold, seed)
                d = SOURCE / "downstream" / rid
                required = ["state.json", "test_report.json", "test_predictions.csv",
                            "test_seal.json", "pairing_proof.json", "best.pt"]
                found = [p for p in required if (d / p).exists()]
                state = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
                report = json.loads((d / "test_report.json").read_text()) if (d / "test_report.json").exists() else {}
                pred_ok = (d / "test_predictions.csv").exists()
                status = "COMPLETE" if state.get("status") == "COMPLETE" and len(found) == len(required) else "INCOMPLETE"
                inventory.append({"arm": arm, "fold": fold, "seed": seed, "run_id": rid,
                                  "status": status, "result_files_found": ";".join(found),
                                  "predictions_found": pred_ok})
                for name in found:
                    source_files.append(d / name)
                if status != "COMPLETE" or not pred_ok:
                    log.append(f"EXCLUDED {rid}: {status}, found={found}")
                    continue
                pred = pd.read_csv(d / "test_predictions.csv", dtype={"y_true": str, "y_pred": str})
                if list(pred.columns) != ["window_id", "dataset", "y_true", "y_pred", "correct"]:
                    raise AssertionError(f"Unexpected prediction schema: {rid}: {list(pred.columns)}")
                if pred.window_id.duplicated().any():
                    raise AssertionError(f"Duplicate window IDs within {rid}")
                if set(pred.window_id) != test_ids:
                    raise AssertionError(f"TEST membership mismatch: {rid}")
                if set(pred.dataset) != {"CWRU", *DATASETS}:
                    raise AssertionError(f"Dataset coverage mismatch: {rid}")
                if not ((pred.y_true == pred.y_pred) == pred.correct).all():
                    raise AssertionError(f"correct column mismatch: {rid}")
                run_predictions[rid] = pred
                for ds in DATASETS:
                    q = pred[pred.dataset == ds]
                    if not set(q.y_true).issubset(CLASSES[ds]) or not set(q.y_pred).issubset(CLASSES[ds]):
                        raise AssertionError(f"Unknown/remapped label: {rid}/{ds}")
                    cm = confusion(q.y_true, q.y_pred, CLASSES[ds])
                    run_cms[(rid, ds)] = cm
                    m = cm_metrics(cm)
                    saved = report["per_dataset_reports"][ds]
                    delta = abs(m["macro_f1"] - float(saved["macro_f1"]))
                    discrepancies.append({"run_id": rid, "dataset": ds,
                                          "new_macro_f1": m["macro_f1"],
                                          "saved_macro_f1": float(saved["macro_f1"]),
                                          "absolute_difference": delta,
                                          "over_1e_6": delta > 1e-6})
                    if cm.tolist() != saved["confusion_matrix"]:
                        raise AssertionError(f"Saved confusion mismatch: {rid}/{ds}")
                    if abs(m["accuracy"] - float(saved["accuracy"])) > 1e-12:
                        raise AssertionError(f"Saved accuracy mismatch: {rid}/{ds}")
                    if int(m["support"].sum()) != len(q):
                        raise AssertionError(f"Support mismatch: {rid}/{ds}")
                    detailed.append({"arm": arm, "fold": fold, "seed": seed,
                        "dataset": DISPLAY[ds], "accuracy": m["accuracy"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                        "macro_f1": m["macro_f1"], "weighted_precision": m["weighted_precision"],
                        "weighted_recall": m["weighted_recall"], "weighted_f1": m["weighted_f1"],
                        "n": m["n"], "macro_auc_ovr": np.nan,
                        "weighted_auc_ovr": np.nan, "micro_auc_ovr": np.nan,
                        "macro_auc_ovo": np.nan, "auc_status": "UNAVAILABLE_NO_CONTINUOUS_SCORES"})
                    for i, label in enumerate(CLASSES[ds]):
                        per_class.append({"arm": arm, "fold": fold, "seed": seed,
                            "dataset": DISPLAY[ds], "numeric_or_original_label": label,
                            "class": SEMANTIC[ds][label], "precision": m["precision"][i],
                            "recall": m["recall"][i], "f1": m["f1"][i],
                            "roc_auc": np.nan, "support": int(m["support"][i]),
                            "positive_n": int(m["support"][i]),
                            "negative_n": int(m["n"] - m["support"][i]),
                            "auc_status": "UNAVAILABLE_NO_CONTINUOUS_SCORES"})
                    stem = f"cm_{ds.lower()}_{arm.lower()}_f{fold}_s{seed}"
                    title = f"{DISPLAY[ds]} — {arm} — Fold {fold} — Seed {seed}"
                    labels = [SEMANTIC[ds][c] for c in CLASSES[ds]]
                    plot_cm(cm, labels, title + " — Raw", CM_IND / f"{stem}_raw.png")
                    plot_cm(cm, labels, title + " — Row-normalized",
                            CM_IND / f"{stem}_normalized.png", True)

    inv = pd.DataFrame(inventory)
    det = pd.DataFrame(detailed)
    pc = pd.DataFrame(per_class)
    inv.to_csv(TABLES / "run_inventory.csv", index=False)
    det.to_csv(TABLES / "classification_detailed.csv", index=False)
    pc.to_csv(TABLES / "per_class_detailed.csv", index=False)
    pd.DataFrame(discrepancies).to_csv(TABLES / "macro_f1_consistency.csv", index=False)

    # Dataset summaries over nine paired fold×seed cells.
    summary_rows = []
    for ds in [*map(DISPLAY.get, DATASETS), "Macro-3"]:
        vals = {}
        for arm in ARMS:
            if ds == "Macro-3":
                cell = det[det.arm == arm].groupby(["fold", "seed"])[
                    ["macro_precision", "macro_recall", "macro_f1", "accuracy", "balanced_accuracy"]].mean()
            else:
                cell = det[(det.arm == arm) & (det.dataset == ds)].set_index(["fold", "seed"])
            for metric in ["macro_precision", "macro_recall", "macro_f1", "accuracy", "balanced_accuracy"]:
                vals[(arm, metric)] = mean_sd(cell[metric])
        summary_rows.append({"dataset": ds,
            "s0_precision_mean": vals[("S0", "macro_precision")][0], "s0_precision_sd": vals[("S0", "macro_precision")][1],
            "s0_recall_mean": vals[("S0", "macro_recall")][0], "s0_recall_sd": vals[("S0", "macro_recall")][1],
            "s0_f1_mean": vals[("S0", "macro_f1")][0], "s0_f1_sd": vals[("S0", "macro_f1")][1],
            "s0_auc_mean": np.nan, "s0_auc_sd": np.nan,
            "s1_precision_mean": vals[("S1", "macro_precision")][0], "s1_precision_sd": vals[("S1", "macro_precision")][1],
            "s1_recall_mean": vals[("S1", "macro_recall")][0], "s1_recall_sd": vals[("S1", "macro_recall")][1],
            "s1_f1_mean": vals[("S1", "macro_f1")][0], "s1_f1_sd": vals[("S1", "macro_f1")][1],
            "s1_auc_mean": np.nan, "s1_auc_sd": np.nan,
            "delta_f1_s1_minus_s0": vals[("S1", "macro_f1")][0] - vals[("S0", "macro_f1")][0],
            "delta_auc_s1_minus_s0": np.nan, "auc_status": "UNAVAILABLE_NO_CONTINUOUS_SCORES",
            "s0_accuracy_mean": vals[("S0", "accuracy")][0], "s0_accuracy_sd": vals[("S0", "accuracy")][1],
            "s1_accuracy_mean": vals[("S1", "accuracy")][0], "s1_accuracy_sd": vals[("S1", "accuracy")][1],
            "s0_balanced_accuracy_mean": vals[("S0", "balanced_accuracy")][0], "s0_balanced_accuracy_sd": vals[("S0", "balanced_accuracy")][1],
            "s1_balanced_accuracy_mean": vals[("S1", "balanced_accuracy")][0], "s1_balanced_accuracy_sd": vals[("S1", "balanced_accuracy")][1]})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "classification_summary_s0_vs_s1.csv", index=False)

    # Paired Macro-3 results.
    paired = []
    for fold in FOLDS:
        for seed in SEEDS:
            f1s = {}
            for arm in ARMS:
                q = det[(det.arm == arm) & (det.fold == fold) & (det.seed == seed)]
                f1s[arm] = float(q.macro_f1.mean())
            paired.append({"fold": fold, "seed": seed, "s0_macro3_f1": f1s["S0"],
                           "s1_macro3_f1": f1s["S1"],
                           "delta_f1_s1_minus_s0": f1s["S1"] - f1s["S0"],
                           "s0_macro3_auc": np.nan, "s1_macro3_auc": np.nan,
                           "delta_auc_s1_minus_s0": np.nan,
                           "auc_status": "UNAVAILABLE_NO_CONTINUOUS_SCORES"})
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(TABLES / "paired_s0_s1.csv", index=False)

    # Per-class summaries.
    for ds in map(DISPLAY.get, DATASETS):
        rows = []
        for cls in pc[pc.dataset == ds]["class"].drop_duplicates():
            row = {"class": cls}
            for arm in ARMS:
                q = pc[(pc.dataset == ds) & (pc["class"] == cls) & (pc.arm == arm)]
                for metric in ["precision", "recall", "f1", "support"]:
                    mu, sd = mean_sd(q[metric])
                    row[f"{arm.lower()}_{metric}_mean"] = mu
                    row[f"{arm.lower()}_{metric}_sd"] = sd
                row[f"{arm.lower()}_auc_mean"] = np.nan
                row[f"{arm.lower()}_auc_sd"] = np.nan
            row["auc_status"] = "UNAVAILABLE_NO_CONTINUOUS_SCORES"
            rows.append(row)
        pd.DataFrame(rows).to_csv(TABLES / f"per_class_{ds.lower()}_summary.csv", index=False)

    mapping = [{"dataset": DISPLAY[ds], "numeric_label": label,
                "semantic_class_name": SEMANTIC[ds][label]}
               for ds in DATASETS for label in CLASSES[ds]]
    pd.DataFrame(mapping).to_csv(TABLES / "class_mapping.csv", index=False)

    # Aggregate count, pooled normalized, and run-mean normalized CMs.
    for ds in DATASETS:
        labels = [SEMANTIC[ds][c] for c in CLASSES[ds]]
        for arm in ARMS:
            cms = [run_cms[(run_id(arm, f, s), ds)] for f in FOLDS for s in SEEDS]
            total = np.sum(cms, axis=0)
            base = f"cm_{ds.lower()}_{arm.lower()}_aggregate"
            title = f"{DISPLAY[ds]} — {arm} — Aggregate of 9 runs"
            plot_cm(total, labels, title + " — Raw (repeated seeds counted)",
                    CM_AGG / f"{base}_raw.png")
            plot_cm(total, labels, title + " — Pooled row-normalized",
                    CM_AGG / f"{base}_normalized.png", True)
            norms = []
            for cm in cms:
                den = cm.sum(1, keepdims=True)
                norms.append(np.divide(cm, den, out=np.zeros_like(cm, dtype=float), where=den > 0))
            mean_norm = np.mean(norms, axis=0)
            plot_cm(mean_norm, labels, title + " — Mean run-normalized",
                    CM_AGG / f"{base}_mean_normalized.png", True)

    # Frozen SSL only retained validation masked-log-STFT MSE, not waveform pairs.
    recon = []
    for fold in FOLDS:
        for seed in SEEDS:
            p = SOURCE / "ssl" / f"ssl_f{fold}_s{seed}" / "completion.json"
            source_files.append(p)
            c = json.loads(p.read_text())
            for ds in DATASETS:
                recon.append({"fold": fold, "seed": seed, "dataset": DISPLAY[ds],
                    "mse": c["best_val_per_dataset_mse"][ds], "nmse": np.nan,
                    "mae": np.nan, "r2": np.nan, "pearson_r": np.nan,
                    "spearman_rho": np.nan, "eligible_for_requested_waveform_summary": False,
                    "scope": "VALIDATION masked-valid-cell MSE on normalized log-STFT patches; not vibration waveform"})
    recon_df = pd.DataFrame(recon)
    recon_df.to_csv(TABLES / "reconstruction_detailed.csv", index=False)
    recon_summary = []
    for ds in [*map(DISPLAY.get, DATASETS), "Macro-3"]:
        if ds == "Macro-3":
            q = recon_df.groupby(["fold", "seed"]).mse.mean()
        else:
            q = recon_df[recon_df.dataset == ds].mse
        mu, sd = mean_sd(q)
        recon_summary.append({"dataset": ds, "mse_mean": mu, "mse_sd": sd,
            "nmse_mean": np.nan, "nmse_sd": np.nan, "mae_mean": np.nan, "mae_sd": np.nan,
            "r2_mean": np.nan, "r2_sd": np.nan, "pearson_r_mean": np.nan, "pearson_r_sd": np.nan,
            "spearman_rho_mean": np.nan, "spearman_rho_sd": np.nan,
            "eligible_for_requested_waveform_summary": False})
    pd.DataFrame(recon_summary).to_csv(TABLES / "reconstruction_summary.csv", index=False)

    # Empty-but-schema-valid AUC tables communicate nonavailability explicitly.
    pd.DataFrame(columns=["arm","fold","seed","dataset","class","auc","positive_n","negative_n","status"]).to_csv(TABLES / "roc_auc_per_class_detailed.csv", index=False)
    pd.DataFrame(columns=["arm","fold","seed","dataset","macro_auc_ovr","weighted_auc_ovr","micro_auc_ovr","macro_auc_ovo","status"]).to_csv(TABLES / "roc_auc_dataset_detailed.csv", index=False)
    summary[["dataset","s0_auc_mean","s0_auc_sd","s1_auc_mean","s1_auc_sd","delta_auc_s1_minus_s0","auc_status"]].to_csv(TABLES / "roc_auc_summary_s0_vs_s1.csv", index=False)

    # Provenance.
    source_files += [REGISTRY / "main_run_registry.csv", REGISTRY / "metric_spec.yaml",
                     REGISTRY / "statistical_analysis_spec.yaml",
                     ROOT / "src/methodology_v2/experiment/metrics.py",
                     ROOT / "src/methodology_v2/experiment/heads.py",
                     ROOT / "scripts/methodology_v2/experiment_executor.py",
                     ROOT / "methodology_v2/part1_audit/dataset_census.csv",
                     Path(__file__).resolve()]
    unique = sorted(set(p.resolve() for p in source_files))
    with (PROV / "source_files.txt").open("w") as f:
        for p in unique:
            f.write(f"{p}\tsha256={sha256(p)}\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (PROV / "extraction_log.txt").write_text("\n".join([
        f"Git commit analysed/current: {commit}", *log,
        f"Complete runs included: {(inv.status == 'COMPLETE').sum()}/18",
        f"Prediction rows included (all four datasets): {sum(len(x) for x in run_predictions.values())}",
        f"Primary-dataset run-level matrices: {len(run_cms)}",
        f"Macro-F1 discrepancies >1e-6: {sum(x['over_1e_6'] for x in discrepancies)}",
        "Continuous TEST scores found: 0 files/columns",
        "Stored sample-level reconstruction target/prediction pairs found: 0",
        "ROC figures generated: 0 (integrity stop: no continuous scores)",
        "Waveform reconstruction figures generated: 0 (integrity stop: no stored waveform pairs)",
    ]) + "\n")
    (PROV / "metric_definitions.md").write_text(METRIC_DEFINITIONS)

    # README uses rounded dissertation-facing tables.
    pstats = {"s0": mean_sd(paired_df.s0_macro3_f1), "s1": mean_sd(paired_df.s1_macro3_f1),
              "delta": mean_sd(paired_df.delta_f1_s1_minus_s0)}
    wins = int((paired_df.delta_f1_s1_minus_s0 > 0).sum())
    losses = int((paired_df.delta_f1_s1_minus_s0 < 0).sum())
    ties = int((paired_df.delta_f1_s1_minus_s0 == 0).sum())
    readme = ["# 100% label final analysis (read-only extraction)", "",
        f"Git commit: `{commit}`. Included 18/18 registered and expected downstream runs (9 S0, 9 S1).",
        "", "## Integrity-limited scope", "",
        "Classification label metrics and confusion matrices were recomputed from the sealed TEST prediction CSVs. ROC/AUC was not computed because no continuous TEST logits/probabilities were retained; hard labels were not misused. Requested vibration-waveform reconstruction metrics/figures were not computed because no sample-level target/reconstruction pairs were retained. The reconstruction CSV preserves only the frozen validation masked-log-STFT MSE and marks it ineligible as waveform reconstruction.",
        "", "## Main classification summary", "",
        "| Dataset | S0 Precision | S0 Recall | S0 F1 | S0 AUC | S1 Precision | S1 Recall | S1 F1 | S1 AUC | ΔF1 | ΔAUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" ]
    for _, r in summary.iterrows():
        readme.append(f"| {r.dataset} | {r.s0_precision_mean:.4f} ± {r.s0_precision_sd:.4f} | {r.s0_recall_mean:.4f} ± {r.s0_recall_sd:.4f} | {r.s0_f1_mean:.4f} ± {r.s0_f1_sd:.4f} | N/A | {r.s1_precision_mean:.4f} ± {r.s1_precision_sd:.4f} | {r.s1_recall_mean:.4f} ± {r.s1_recall_sd:.4f} | {r.s1_f1_mean:.4f} ± {r.s1_f1_sd:.4f} | N/A | {r.delta_f1_s1_minus_s0:.4f} | N/A |")
    readme += ["", "## Paired Macro-3", "",
        f"S0 {fmt(pstats['s0'])}; S1 {fmt(pstats['s1'])}; paired difference {fmt(pstats['delta'])}; S1 wins/S0 wins/ties = {wins}/{losses}/{ties}.",
        "", "The frozen registry specifies an exact two-sided paired sign-flip permutation test (512 flips) at 100% labels. No new test was run here.",
        "", "## Outputs", "",
        "See `tables/`, `figures/confusion_matrices/`, and `provenance/`. ROC and reconstruction figure directories are intentionally empty and the integrity stops are documented in provenance."]
    (OUT / "README.md").write_text("\n".join(readme) + "\n")


METRIC_DEFINITIONS = r"""# Metric definitions and eligibility

## Classification

All metrics operate per `arm × fold × seed × dataset` on sealed TEST hard-label predictions. Class order is frozen by `metric_spec.yaml` and `heads.py`; absent/undefined class precision, recall, and F1 are zero and classes are never dropped.

- Accuracy = `sum_k TP_k / N`.
- Balanced accuracy = macro recall = `mean_k TP_k/(TP_k+FN_k)`.
- Per-class precision = `TP/(TP+FP)`; recall = `TP/(TP+FN)`; F1 = harmonic mean of precision and recall.
- Macro metrics are unweighted arithmetic means across the frozen class list.
- Weighted metrics weight per-class metrics by TEST support.
- Macro-3 first calculates a metric separately for JNU, HIT, and MaFaulDa in each paired fold×seed cell, then takes their unweighted mean. Summary SD is sample SD (`ddof=1`) over nine cells.

Aggregate raw confusion matrices sum nine matrices. Each physical fold TEST membership is repeated across three seeds, so counts include each fold's samples three times. Pooled row-normalized matrices normalize these sums. Mean-normalized matrices normalize each run by true-class row and then average the nine run matrices, giving runs equal weight.

## ROC/AUC

Unavailable. The sealed TEST prediction artifacts contain hard labels only; no logits, probabilities, or other continuous decision scores were retained. ROC is never calculated from hard predicted labels. Re-running checkpoint inference would contradict the executor's documented single sealed TEST evaluation and is outside this read-only extraction.

## Reconstruction

The frozen definition that exists is only `MacroDomainReconMSE`: for each dataset, mean of per-window MSE over masked valid normalized log-STFT cells; then an equal-dataset macro mean. The stored values are validation-set checkpoint metrics, not vibration-waveform TEST reconstructions.

No frozen NMSE definition exists. No NMSE is proposed or calculated because no stored target/reconstruction waveform pairs exist. MAE, R², Pearson r, and Spearman rho are likewise unavailable. Consequently there is no defensible window-vs-concatenated choice, scale inversion, constant-signal handling, or NaN correlation count to report; those calculations were stopped rather than invented.
"""

if __name__ == "__main__":
    main()
