"""Final, one-time test evaluation after calibration/thresholds are frozen.
Creates baseline 0.5 metrics, proposed referral metrics, confusion-matrix images,
and requested comparison charts/tables. No parameter selection happens here.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

DATA_ROOT = Path("./iccad-official")
OUTPUT_DIR = Path("./outputs/final_results")
BENCHMARKS = [1, 2, 3, 4, 5]
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPS = 1e-7


def load_test(benchmark):
    folder = DATA_ROOT / f"iccad{benchmark}" / "test"
    ds = tf.keras.utils.image_dataset_from_directory(
        str(folder), labels="inferred", label_mode="binary", color_mode="grayscale",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    classes = ds.class_names
    hs_idx = next((i for i, n in enumerate(classes) if "nhs" not in n.lower()), None)
    if hs_idx is None: raise ValueError(f"Cannot find hotspot class in {classes}")
    paths = list(ds.file_paths)
    xs, ys = [], []
    for x, y in ds:
        xs.append((tf.cast(x, tf.float32) / 255.0).numpy())
        ys.extend(y.numpy().astype(int).ravel().tolist())
    y_class = np.asarray(ys, dtype=int)
    y = (1-y_class) if hs_idx == 0 else y_class
    return np.concatenate(xs), y.astype(int), paths, classes, hs_idx


def compute_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    n_hot = int((y == 1).sum())
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
            "specificity": float(tn/(tn+fp)) if tn+fp else float("nan"),
            "f1_score": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float((recall_score(y,pred,zero_division=0) + (tn/(tn+fp) if tn+fp else 0))/2),
            "total_images": int(len(y)), "total_hotspots": n_hot,
            "baseline_missed_hotspots": int(fn),
            "baseline_missed_hotspot_rate": float(fn/n_hot) if n_hot else float("nan"),
            "false_negative_rate": float(fn/n_hot) if n_hot else float("nan")}


def plot_cm(cm, title, path, labels=("Non-hotspot", "Hotspot")):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(2), yticks=np.arange(2), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted label", ylabel="True label", title=title)
    cutoff = cm.max()/2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cutoff else "black")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    config_path = OUTPUT_DIR / "final_calibration_and_thresholds.json"
    if not config_path.exists(): raise FileNotFoundError("Run 02_calibrate_and_select_thresholds.py first.")
    config = json.loads(config_path.read_text())
    low, high = config["selected_low_threshold"], config["selected_high_threshold"]
    baseline_rows, proposed_rows, prediction_frames = [], [], []
    all_y, all_baseline_pred, all_proposed_operational_pred = [], [], []

    for b in BENCHMARKS:
        x, y, paths, classes, hs_idx = load_test(b)
        model = tf.keras.models.load_model(OUTPUT_DIR / f"model_benchmark_{b}.keras")
        raw = model.predict(x, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        hotspot_logits = -raw if hs_idx == 0 else raw
        p_raw = tf.math.sigmoid(hotspot_logits).numpy()
        t = float(config["temperatures_by_benchmark"][str(b)])
        p_cal = tf.math.sigmoid(hotspot_logits / t).numpy()
        baseline_pred = (p_raw >= 0.5).astype(int)
        base = compute_metrics(y, baseline_pred); base["benchmark"] = b
        baseline_rows.append(base)

        bucket = np.where(p_cal <= low, "safe", np.where(p_cal >= high, "risky", "needs_check"))
        hs_mask = y == 1
        missed = int(np.sum((bucket == "safe") & hs_mask))
        n_hs = int(hs_mask.sum())
        total = len(y)
        n_safe, n_risky, n_check = int((bucket=="safe").sum()), int((bucket=="risky").sum()), int((bucket=="needs_check").sum())
        # For a binary operational confusion matrix only: risky OR referred => not passed as safe.
        # This is NOT a verified final classification for needs_check cases.
        operational_pred = (bucket != "safe").astype(int)
        prop_cm = confusion_matrix(y, operational_pred, labels=[0,1])
        prop = {"benchmark": b, "low_threshold": low, "high_threshold": high,
                "total_images": total, "total_hotspots": n_hs,
                "n_safe": n_safe, "n_risky": n_risky, "n_needs_check": n_check,
                "safe_rate": n_safe/total if total else 0,
                "risk_rate": n_risky/total if total else 0,
                "referral_rate": n_check/total if total else 0,
                "screening_rate": (n_safe+n_risky)/total if total else 0,
                "missed_hotspots_marked_safe": missed,
                "missed_hotspot_rate": missed/n_hs if n_hs else float("nan"),
                "operational_TN": int(prop_cm[0,0]), "operational_FP": int(prop_cm[0,1]),
                "operational_FN": int(prop_cm[1,0]), "operational_TP": int(prop_cm[1,1]),
                "operational_false_negative_rate": float(prop_cm[1,0]/n_hs) if n_hs else float("nan")}
        proposed_rows.append(prop)
        df = pd.DataFrame({"file_path": paths, "true_label": y, "raw_hotspot_probability": p_raw,
                           "temperature_T": t, "calibrated_hotspot_probability": p_cal, "bucket": bucket})
        df.to_csv(OUTPUT_DIR / f"test_predictions_benchmark_{b}.csv", index=False)
        prediction_frames.append(df.assign(benchmark=b))
        all_y.extend(y.tolist()); all_baseline_pred.extend(baseline_pred.tolist()); all_proposed_operational_pred.extend(operational_pred.tolist())
        plot_cm(confusion_matrix(y, baseline_pred, labels=[0,1]), f"ICCAD-{b}: Baseline (0.5 threshold)",
                OUTPUT_DIR / f"confusion_matrix_baseline_iccad{b}.png")
        plot_cm(prop_cm, f"ICCAD-{b}: Proposed operational (safe vs not-safe)",
                OUTPUT_DIR / f"confusion_matrix_proposed_iccad{b}.png")

    baseline_df = pd.DataFrame(baseline_rows)
    proposed_df = pd.DataFrame(proposed_rows)
    baseline_df.to_csv(OUTPUT_DIR / "test_baseline_metrics.csv", index=False)
    proposed_df.to_csv(OUTPUT_DIR / "test_proposed_referral_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUTPUT_DIR / "test_all_predictions.csv", index=False)

    # Pooled confusion matrices across all test images, also useful in the report.
    plot_cm(confusion_matrix(all_y, all_baseline_pred, labels=[0,1]), "All ICCAD sets: Baseline (0.5)", OUTPUT_DIR / "confusion_matrix_baseline_all.png")
    plot_cm(confusion_matrix(all_y, all_proposed_operational_pred, labels=[0,1]), "All ICCAD sets: Proposed operational", OUTPUT_DIR / "confusion_matrix_proposed_all.png")

    comp = baseline_df[["benchmark", "baseline_missed_hotspot_rate", "false_negative_rate", "FN", "TP", "total_hotspots"]].merge(
        proposed_df[["benchmark", "missed_hotspot_rate", "referral_rate", "safe_rate", "risk_rate", "screening_rate", "missed_hotspots_marked_safe", "n_needs_check"]], on="benchmark")
    comp["missed_rate_improvement_percentage_points"] = (comp["baseline_missed_hotspot_rate"] - comp["missed_hotspot_rate"])*100
    comp.to_csv(OUTPUT_DIR / "test_baseline_vs_proposed_comparison.csv", index=False)

    # Requested bar chart: baseline vs proposed missed hotspot rate for each ICCAD benchmark.
    x = np.arange(len(comp)); width = 0.36
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x-width/2, comp["baseline_missed_hotspot_rate"]*100, width, label="Baseline (0.5)")
    ax.bar(x+width/2, comp["missed_hotspot_rate"]*100, width, label=f"Proposed ({low:.2f}/{high:.2f})")
    ax.set_xticks(x); ax.set_xticklabels([f"ICCAD-{b}" for b in comp["benchmark"]])
    ax.set_ylabel("Missed hotspot rate (%)"); ax.set_title("Baseline vs Proposed: Missed Hotspot Rate")
    ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT_DIR/"chart_missed_hotspots_baseline_vs_proposed.png", dpi=160); plt.close(fig)

    # Requested threshold trade-off plot. A point is one candidate threshold pair,
    # pooled over all benchmarks. Horizontal 5% line marks eligibility boundary.
    val = pd.read_csv(OUTPUT_DIR / "validation_threshold_selection.csv")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.axhline(5, linestyle="--", label="5% maximum validation missed-rate constraint")
    # Plot validation trade-off, because test must not be used to select thresholds.
    for _, r in val.iterrows():
        ax.scatter(r["pooled_referral_rate"]*100, r["worst_benchmark_missed_hotspot_rate"]*100, s=65)
        ax.annotate(f"{r['low_threshold']:.2f}/{r['high_threshold']:.2f}",
                    (r["pooled_referral_rate"]*100, r["worst_benchmark_missed_hotspot_rate"]*100),
                    xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("Validation referral rate (%)"); ax.set_ylabel("Worst-benchmark missed hotspot rate (%)")
    ax.set_title("Validation Trade-off: Missed Hotspots vs Referral Workload")
    ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT_DIR/"chart_validation_missed_vs_referral.png", dpi=160); plt.close(fig)

    # Test trade-off is reported descriptively after the choice is frozen, but not used to choose it.
    test_trade = proposed_df[["benchmark", "referral_rate", "missed_hotspot_rate"]]
    test_trade.to_csv(OUTPUT_DIR / "test_tradeoff_points.csv", index=False)
    print("Final test evaluation complete. Thresholds were loaded from the frozen validation config.")
    print(f"Selected thresholds: {low:.2f}/{high:.2f}")
    print("Key outputs: test_baseline_metrics.csv, test_proposed_referral_metrics.csv,")
    print("test_baseline_vs_proposed_comparison.csv, confusion-matrix PNGs, and comparison charts.")

if __name__ == "__main__": main()
