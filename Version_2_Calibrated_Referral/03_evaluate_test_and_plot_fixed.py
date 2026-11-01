"""Final, one-time test evaluation after calibration/thresholds are frozen.

Reads the trained models from outputs/calib_validation, the selected threshold
configuration from Module 02, evaluates the untouched test folders, and saves
metrics/tables/plots to outputs/final_results.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)

# Paths are relative to ~/aiml/hotspot_pipeline_modules when run from that folder.
DATA_ROOT = Path("../iccad_train_validation_split/iccad-official")
CALIBRATION_DIR = Path("./outputs/calib_validation")
MODEL_DIR = CALIBRATION_DIR
OUTPUT_DIR = Path("./outputs/final_results")

BENCHMARKS = [1, 2, 3, 4, 5]
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPS = 1e-7

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_test(benchmark):
    folder = DATA_ROOT / f"iccad{benchmark}" / "test"
    if not folder.exists():
        raise FileNotFoundError(f"Missing test folder: {folder}")
    ds = tf.keras.utils.image_dataset_from_directory(
        str(folder), labels="inferred", label_mode="binary",
        color_mode="grayscale", image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, shuffle=False
    )
    classes = ds.class_names
    hs_matches = [i for i, name in enumerate(classes) if "nhs" not in name.lower()]
    if len(hs_matches) != 1:
        raise ValueError(f"Cannot identify hotspot class from {classes}; expected hs and nhs.")
    hs_idx = hs_matches[0]
    xs, y_class, paths = [], [], list(ds.file_paths)
    for x, y in ds:
        xs.append((tf.cast(x, tf.float32) / 255.0).numpy())
        y_class.extend(y.numpy().astype(int).ravel().tolist())
    y_class = np.asarray(y_class, dtype=int)
    y_hotspot = (1 - y_class) if hs_idx == 0 else y_class
    return np.concatenate(xs), y_hotspot.astype(int), paths, hs_idx


def get_hotspot_logits(model, x, hs_idx):
    raw = model.predict(x, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    # Module 01's linear output logit maps to class index 1.
    return -raw if hs_idx == 0 else raw


def binary_metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    n_hotspots = int((y == 1).sum())
    return {
        "total_images": n,
        "total_hotspots": n_hotspots,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(((tp / (tp + fn)) if (tp + fn) else 0.0
                                    + (tn / (tn + fp)) if (tn + fp) else 0.0) / 2),
        "missed_hotspots": int(fn),
        "missed_hotspot_rate_pct": float(100 * fn / n_hotspots) if n_hotspots else np.nan,
        "false_negative_rate_pct": float(100 * fn / n_hotspots) if n_hotspots else np.nan,
    }


def referral_metrics(y, p, low, high):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    safe = p <= low
    risky = p >= high
    needs_check = ~(safe | risky)
    n = len(y)
    n_hs = int((y == 1).sum())
    missed = int(np.sum((y == 1) & safe))
    # Operational binary matrix: safe=negative/pass; risky or needs_check=positive/not passed as safe.
    operational_pred = (~safe).astype(int)
    m = binary_metrics(y, operational_pred)
    m.update({
        "safe_count": int(safe.sum()),
        "risky_count": int(risky.sum()),
        "needs_check_count": int(needs_check.sum()),
        "safe_rate_pct": float(100 * safe.mean()) if n else np.nan,
        "risky_rate_pct": float(100 * risky.mean()) if n else np.nan,
        "referral_rate_pct": float(100 * needs_check.mean()) if n else np.nan,
        "screening_rate_pct": float(100 * (safe | risky).mean()) if n else np.nan,
        "missed_hotspots_marked_safe": missed,
        "missed_hotspot_rate_pct": float(100 * missed / n_hs) if n_hs else np.nan,
    })
    return m, operational_pred, safe, risky, needs_check


def save_cm(y, pred, path, title):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted label (0=non-hotspot, 1=hotspot)")
    ax.set_ylabel("Actual label (0=non-hotspot, 1=hotspot)")
    ax.set_xticks([0, 1], ["Non-hotspot", "Hotspot"])
    ax.set_yticks([0, 1], ["Non-hotspot", "Hotspot"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    config_path = CALIBRATION_DIR / "selected_threshold.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cannot find {config_path}.\n"
            "First run Module 02 successfully. It must select an eligible threshold "
            "and create selected_threshold.json. If no candidate meets the 7.5% "
            "constraint, Module 02 will not create a selected configuration."
        )
    config = json.loads(config_path.read_text())
    low = float(config["low_threshold"])
    high = float(config["high_threshold"])

    # Load per-benchmark temperatures written by Module 02.
    temp_path = CALIBRATION_DIR / "validation_temperature_summary.csv"
    if not temp_path.exists():
        raise FileNotFoundError(f"Missing temperature file: {temp_path}; rerun Module 02.")
    temp_df = pd.read_csv(temp_path)
    if not {"benchmark", "temperature_T"}.issubset(temp_df.columns):
        raise ValueError(f"Unexpected columns in {temp_path}: {list(temp_df.columns)}")
    temperatures = {int(r["benchmark"]): float(r["temperature_T"])
                    for _, r in temp_df.iterrows()}

    baseline_rows, proposed_rows = [], []
    all_y, all_base, all_operational = [], [], []
    chart_rows = []

    for benchmark in BENCHMARKS:
        model_path = MODEL_DIR / f"model_benchmark_{benchmark}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model: {model_path}; run Module 01.")
        if benchmark not in temperatures:
            raise ValueError(f"No fitted temperature for ICCAD-{benchmark} in {temp_path}")

        x, y, paths, hs_idx = load_test(benchmark)
        model = tf.keras.models.load_model(model_path)
        logits = get_hotspot_logits(model, x, hs_idx)
        p_base = tf.math.sigmoid(logits).numpy()
        p_cal = tf.math.sigmoid(logits / temperatures[benchmark]).numpy()
        pred_base = (p_base >= 0.5).astype(int)

        base_metrics = binary_metrics(y, pred_base)
        baseline_rows.append({"benchmark": benchmark, **base_metrics})
        prop_metrics, pred_operational, safe, risky, needs_check = referral_metrics(
            y, p_cal, low, high
        )
        proposed_rows.append({
            "benchmark": benchmark, "low_threshold": low, "high_threshold": high,
            "temperature_T": temperatures[benchmark], **prop_metrics
        })
        chart_rows.append({
            "benchmark": benchmark,
            "baseline_missed_hotspot_rate_pct": base_metrics["missed_hotspot_rate_pct"],
            "proposed_missed_hotspot_rate_pct": prop_metrics["missed_hotspot_rate_pct"],
            "referral_rate_pct": prop_metrics["referral_rate_pct"],
        })

        save_cm(y, pred_base, OUTPUT_DIR / f"confusion_matrix_baseline_iccad{benchmark}.png",
                f"Baseline 0.5 — ICCAD-{benchmark}")
        save_cm(y, pred_operational,
                OUTPUT_DIR / f"confusion_matrix_proposed_iccad{benchmark}.png",
                f"Proposed operational — ICCAD-{benchmark}")

        pd.DataFrame({
            "file_path": paths, "true_label": y,
            "baseline_hotspot_probability": p_base,
            "calibrated_hotspot_probability": p_cal,
            "bucket": np.where(safe, "safe", np.where(risky, "risky", "needs_check")),
        }).to_csv(OUTPUT_DIR / f"test_predictions_iccad{benchmark}.csv", index=False)

        all_y.extend(y.tolist())
        all_base.extend(pred_base.tolist())
        all_operational.extend(pred_operational.tolist())

    # Per-benchmark metrics.
    baseline_df = pd.DataFrame(baseline_rows)
    proposed_df = pd.DataFrame(proposed_rows)
    baseline_df.to_csv(OUTPUT_DIR / "baseline_test_metrics_by_benchmark.csv", index=False)
    proposed_df.to_csv(OUTPUT_DIR / "proposed_test_metrics_by_benchmark.csv", index=False)

    # Overall pooled metrics from counts, not average of percentages.
    all_y = np.asarray(all_y, dtype=int)
    all_base = np.asarray(all_base, dtype=int)
    all_operational = np.asarray(all_operational, dtype=int)
    overall_base = binary_metrics(all_y, all_base)
    overall_prop = binary_metrics(all_y, all_operational)
    overall_prop.update({
        "low_threshold": low, "high_threshold": high,
        "interpretation": "Operational: only images marked safe count as negative/pass."
    })
    pd.DataFrame([overall_base]).to_csv(OUTPUT_DIR / "baseline_test_metrics_overall.csv", index=False)
    pd.DataFrame([overall_prop]).to_csv(OUTPUT_DIR / "proposed_test_metrics_overall.csv", index=False)

    save_cm(all_y, all_base, OUTPUT_DIR / "confusion_matrix_baseline_all.png",
            "Baseline 0.5 — all ICCAD benchmarks")
    save_cm(all_y, all_operational, OUTPUT_DIR / "confusion_matrix_proposed_all.png",
            "Proposed operational — all ICCAD benchmarks")

    chart_df = pd.DataFrame(chart_rows)
    chart_df.to_csv(OUTPUT_DIR / "missed_hotspot_comparison_by_benchmark.csv", index=False)
    x = np.arange(len(chart_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width/2, chart_df["baseline_missed_hotspot_rate_pct"], width,
           label="Baseline (0.5)")
    ax.bar(x + width/2, chart_df["proposed_missed_hotspot_rate_pct"], width,
           label=f"Proposed referral ({int(low*100)}-{int(high*100)})")
    ax.set_xticks(x, [f"ICCAD-{b}" for b in chart_df["benchmark"]])
    ax.set_ylabel("Missed-hotspot rate (%)")
    ax.set_title("Baseline vs Proposed Missed-Hotspot Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_missed_hotspots_baseline_vs_proposed.png", dpi=160)
    plt.close(fig)

    # Validation trade-off chart: missed-hotspot rate (not pooled rate) vs referral rate.
    # Plot all requested threshold candidates, marking the 7.5% maximum constraint.
    validation_summary = CALIBRATION_DIR / "threshold_metrics_pooled_summary.csv"
    if validation_summary.exists():
        vdf = pd.read_csv(validation_summary)

        # Use the worst-benchmark missed-hotspot rate for the safety constraint,
        # because eligibility was defined per benchmark. The pooled referral rate
        # remains the workload measure on the x-axis.
        required_cols = {
            "threshold_range", "pooled_referral_rate_pct",
            "worst_benchmark_missed_hotspot_rate_pct"
        }
        missing = required_cols - set(vdf.columns)
        if missing:
            raise ValueError(
                f"{validation_summary} is missing required columns: {sorted(missing)}"
            )

        # Display in the requested order from narrowest referral band to widest.
        display_order = ["35-65", "30-70", "20-80", "15-85", "10-90"]
        vdf["threshold_range"] = vdf["threshold_range"].astype(str)
        vdf = vdf.set_index("threshold_range").reindex(
            [r for r in display_order if r in set(vdf["threshold_range"])]
        ).reset_index()

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(
            vdf["pooled_referral_rate_pct"],
            vdf["worst_benchmark_missed_hotspot_rate_pct"]
        )
        for _, row in vdf.iterrows():
            ax.annotate(
                str(row["threshold_range"]),
                (row["pooled_referral_rate_pct"],
                 row["worst_benchmark_missed_hotspot_rate_pct"]),
                xytext=(6, 5), textcoords="offset points"
            )

        ax.axhline(
            7.5, linestyle="--",
            label="Maximum missed-hotspot constraint (7.5%)"
        )
        ax.set_xlabel("Referral rate (%)")
        ax.set_ylabel("Missed-hotspot rate (%)")
        ax.set_title("Missed-Hotspot Rate vs Referral Rate (Validation Data)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "chart_validation_missed_vs_referral.png", dpi=160)
        plt.close(fig)

    print(f"Selected threshold: {low:.2f}-{high:.2f}")
    print(f"Baseline overall test missed-hotspot rate: {overall_base['missed_hotspot_rate_pct']:.2f}%")
    print(f"Proposed overall test missed-hotspot rate: {overall_prop['missed_hotspot_rate_pct']:.2f}%")
    print(f"Saved final test results and plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
