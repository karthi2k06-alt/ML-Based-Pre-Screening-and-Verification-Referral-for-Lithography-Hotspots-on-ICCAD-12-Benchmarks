"""Fit temperature scaling and select referral thresholds using validation data only.

Runs one CNN per ICCAD benchmark, fits a positive temperature T on that
benchmark's validation logits, evaluates the requested threshold ranges, and
selects the eligible candidate with the lowest pooled referral rate.

Constraint: worst-benchmark validation missed-hotspot rate <= 7.5%.
No test images are loaded by this script.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

# Run from ~/aiml/hotspot_pipeline_modules
DATA_ROOT = Path("../iccad_train_validation_split/iccad-official")
MODEL_DIR = Path("./outputs/calib_validation")  # must match Module 01 OUTPUT_DIR
OUTPUT_DIR = Path("./outputs/calib_validation")

BENCHMARKS = [1, 2, 3, 4, 5]
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
MAX_ALLOWED_MISSED_RATE = 0.0755
EPS = 1e-7

# Candidate thresholds, interpreted as P(hotspot):
# P <= low: safe; P >= high: risky; between: needs_check.
THRESHOLD_RANGES = [
    (0.10, 0.90),
    (0.15, 0.85),
    (0.20, 0.80),
    (0.32, 0.70),
    (0.35, 0.65),
]


def load_validation(benchmark):
    folder = DATA_ROOT / f"iccad{benchmark}" / "validation"
    if not folder.exists():
        raise FileNotFoundError(f"Missing validation folder: {folder}")
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
    xs, ys_class = [], []
    for x, y in ds:
        xs.append((tf.cast(x, tf.float32) / 255.0).numpy())
        ys_class.extend(y.numpy().astype(int).ravel().tolist())
    y_class = np.asarray(ys_class, dtype=int)
    y_hotspot = (1 - y_class) if hs_idx == 0 else y_class
    return np.concatenate(xs), y_hotspot.astype(int), classes, hs_idx


def get_hotspot_logits(model, x, hs_idx):
    raw = model.predict(x, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    # Training metadata/model convention: raw logit maps to class index 1.
    return -raw if hs_idx == 0 else raw


def fit_temperature(logits, y_true):
    y = np.asarray(y_true, dtype=int)

    def objective(log_t):
        t = float(np.exp(log_t))
        p = tf.math.sigmoid(logits / t).numpy()
        return log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x)), float(objective(result.x))


def summarize_baseline(y_true, p):
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(p) >= 0.5
    hs = y_true == 1
    nhs = ~hs
    tp = int(np.sum(hs & pred))
    fn = int(np.sum(hs & ~pred))
    fp = int(np.sum(nhs & pred))
    tn = int(np.sum(nhs & ~pred))
    n_hs = int(hs.sum())
    return {
        "total_images": len(y_true), "actual_hotspots": n_hs,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "baseline_missed_hotspot_rate_pct": 100 * fn / n_hs if n_hs else np.nan,
        "baseline_false_negative_rate_pct": 100 * fn / n_hs if n_hs else np.nan,
        "baseline_accuracy_pct": 100 * (tp + tn) / len(y_true) if len(y_true) else np.nan,
    }


def summarize_referral(y_true, p, low, high):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    safe = p <= low
    risky = p >= high
    check = ~(safe | risky)
    n_hs = int((y_true == 1).sum())
    missed = int(np.sum((y_true == 1) & safe))
    n = len(y_true)
    return {
        "total_images": n,
        "actual_hotspots": n_hs,
        "safe_count": int(safe.sum()),
        "risky_count": int(risky.sum()),
        "needs_check_count": int(check.sum()),
        "safe_rate_pct": 100 * safe.mean() if n else np.nan,
        "risky_rate_pct": 100 * risky.mean() if n else np.nan,
        "referral_rate_pct": 100 * check.mean() if n else np.nan,
        "missed_hotspots": missed,
        "missed_hotspot_rate_pct": 100 * missed / n_hs if n_hs else np.nan,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_rows, temp_rows, candidate_rows = [], [], []
    prediction_frames = []

    for benchmark in BENCHMARKS:
        model_path = MODEL_DIR / f"model_benchmark_{benchmark}.keras"
        meta_path = MODEL_DIR / f"model_benchmark_{benchmark}_metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing model/metadata for ICCAD-{benchmark}.\n"
                f"Expected:\n  {model_path}\n  {meta_path}\n"
                "Run 01_train_models.py first and ensure its OUTPUT_DIR matches MODEL_DIR."
            )

        x_val, y_val, classes, hs_idx = load_validation(benchmark)
        model = tf.keras.models.load_model(model_path)
        logits = get_hotspot_logits(model, x_val, hs_idx)
        p_raw = tf.math.sigmoid(logits).numpy()
        base = summarize_baseline(y_val, p_raw)
        baseline_rows.append({"benchmark": benchmark, **base})

        temperature, calibrated_nll = fit_temperature(logits, y_val)
        p_cal = tf.math.sigmoid(logits / temperature).numpy()
        raw_nll = log_loss(y_val, np.clip(p_raw, EPS, 1-EPS), labels=[0, 1])
        temp_rows.append({
            "benchmark": benchmark, "temperature_T": temperature,
            "validation_images": len(y_val), "validation_hotspots": int(y_val.sum()),
            "validation_NLL_before": raw_nll, "validation_NLL_after": calibrated_nll,
        })

        prediction_frames.append(pd.DataFrame({
            "benchmark": benchmark, "true_label": y_val, "hotspot_logit": logits,
            "raw_hotspot_probability": p_raw, "calibrated_hotspot_probability": p_cal,
        }))

        for low, high in THRESHOLD_RANGES:
            stats = summarize_referral(y_val, p_cal, low, high)
            candidate_rows.append({
                "benchmark": benchmark,
                "threshold_range": f"{int(low*100)}-{int(high*100)}",
                "low_threshold": low, "high_threshold": high,
                "temperature_T": temperature, **stats,
            })
        print(f"ICCAD-{benchmark}: baseline MHR={base['baseline_missed_hotspot_rate_pct']:.3f}% | T={temperature:.5f}")

    baseline_df = pd.DataFrame(baseline_rows)
    temp_df = pd.DataFrame(temp_rows)
    per_df = pd.DataFrame(candidate_rows)
    pooled_rows = []
    for (label, low, high), group in per_df.groupby(
        ["threshold_range", "low_threshold", "high_threshold"], sort=False
    ):
        n_images = int(group["total_images"].sum())
        n_hotspots = int(group["actual_hotspots"].sum())
        n_missed = int(group["missed_hotspots"].sum())
        n_referrals = int(group["needs_check_count"].sum())
        worst = float(group["missed_hotspot_rate_pct"].max())
        pooled_rows.append({
            "threshold_range": label, "low_threshold": low, "high_threshold": high,
            "pooled_images": n_images, "pooled_actual_hotspots": n_hotspots,
            "pooled_missed_hotspots": n_missed,
            "pooled_missed_hotspot_rate_pct": 100*n_missed/n_hotspots if n_hotspots else np.nan,
            "pooled_referral_rate_pct": 100*n_referrals/n_images if n_images else np.nan,
            "worst_benchmark_missed_hotspot_rate_pct": worst,
            "meets_7_5pct_worst_benchmark_constraint": bool(worst <= 100*MAX_ALLOWED_MISSED_RATE),
        })
    pooled_df = pd.DataFrame(pooled_rows)
    eligible = pooled_df[pooled_df["meets_7_5pct_worst_benchmark_constraint"]].copy()

    baseline_df.to_csv(OUTPUT_DIR / "baseline_validation_metrics_by_benchmark.csv", index=False)
    temp_df.to_csv(OUTPUT_DIR / "validation_temperature_summary.csv", index=False)
    per_df.to_csv(OUTPUT_DIR / "threshold_metrics_per_benchmark.csv", index=False)
    pooled_df.to_csv(OUTPUT_DIR / "threshold_metrics_pooled_summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "validation_predictions_all_benchmarks.csv", index=False
    )

    statement_path = OUTPUT_DIR / "threshold_selection_justification.txt"
    if eligible.empty:
        statement = (
            "THRESHOLD SELECTION RESULT\n"
            "No candidate threshold range satisfies the maximum 7.5% worst-benchmark "
            "validation missed-hotspot constraint. No range has been selected. "
            "Do not tune on test data. Add/adjust candidate ranges or improve the model "
            "using training/validation data, then rerun this module.\n"
        )
    else:
        # Select minimum pooled referral rate; tie-break by lower worst-benchmark MHR,
        # then lower pooled MHR.
        selected = eligible.sort_values(
            ["pooled_referral_rate_pct", "worst_benchmark_missed_hotspot_rate_pct",
             "pooled_missed_hotspot_rate_pct"],
            ascending=True
        ).iloc[0]
        statement = (
            "THRESHOLD SELECTION JUSTIFICATION (VALIDATION DATA ONLY)\n\n"
            f"Selected threshold range: {selected['threshold_range']} "
            f"(low={selected['low_threshold']:.2f}, high={selected['high_threshold']:.2f}).\n"
            f"Worst-benchmark validation missed-hotspot rate: "
            f"{selected['worst_benchmark_missed_hotspot_rate_pct']:.2f}%.\n"
            f"Pooled validation missed-hotspot rate: "
            f"{selected['pooled_missed_hotspot_rate_pct']:.2f}%.\n"
            f"Pooled validation referral rate: {selected['pooled_referral_rate_pct']:.2f}%.\n"
            "Reason: this candidate satisfies the maximum 7.5% worst-benchmark validation "
            "missed-hotspot constraint and has the lowest pooled referral rate among "
            "eligible candidates, thereby reducing verification workload while meeting "
            "the selected constraint.\n\n"
            "Temperature scaling: a separate positive temperature T was optimized for "
            "each benchmark by minimizing validation binary cross-entropy (negative "
            "log-likelihood). The fitted T values are listed in "
            "validation_temperature_summary.csv; they were optimized, not manually "
            "chosen arbitrarily.\n\n"
            "Important: this is a validation-based selection. Freeze the selected "
            "temperatures and threshold range before evaluating the untouched test set.\n"
        )
        (OUTPUT_DIR / "selected_threshold.json").write_text(json.dumps({
            "threshold_range": str(selected["threshold_range"]),
            "low_threshold": float(selected["low_threshold"]),
            "high_threshold": float(selected["high_threshold"]),
            "max_allowed_worst_benchmark_missed_rate": MAX_ALLOWED_MISSED_RATE,
            "pooled_missed_hotspot_rate_pct": float(selected["pooled_missed_hotspot_rate_pct"]),
            "worst_benchmark_missed_hotspot_rate_pct": float(selected["worst_benchmark_missed_hotspot_rate_pct"]),
            "pooled_referral_rate_pct": float(selected["pooled_referral_rate_pct"]),
        }, indent=2))
    statement_path.write_text(statement)

    print("\nBASELINE VALIDATION MISSED-HOTSPOT RATE (raw P >= 0.5 predicts hotspot)")
    print(baseline_df[["benchmark", "actual_hotspots", "FN",
                       "baseline_missed_hotspot_rate_pct"]].to_string(index=False))
    print("\nTHRESHOLD SUMMARY (validation; calibrated probabilities)")
    print(pooled_df[["threshold_range", "pooled_missed_hotspot_rate_pct",
                     "worst_benchmark_missed_hotspot_rate_pct",
                     "pooled_referral_rate_pct",
                     "meets_7_5pct_worst_benchmark_constraint"]].to_string(index=False))
    if eligible.empty:
        print("\nNo candidate satisfies the 7.5% worst-benchmark constraint; no selection made.")
    else:
        print("\n" + statement)
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
