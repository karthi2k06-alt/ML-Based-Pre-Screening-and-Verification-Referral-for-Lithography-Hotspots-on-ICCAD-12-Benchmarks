"""Validation-only threshold exploration for ICCAD hotspot referral.

Reports baseline missed-hotspot rate at raw probability threshold 0.5 and
calibrated referral metrics for candidate low/high threshold pairs.
Does NOT load or evaluate the test split and does NOT enforce the 5% limit
by stopping; it reports eligibility for every candidate.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

# Run this script from ~/aiml/hotspot_pipeline_modules
DATA_ROOT = Path("../iccad_train_validation_split/iccad-official")
MODEL_DIR = Path("./outputs/calib_validation")  # must match Module 01 OUTPUT_DIR
OUTPUT_DIR = Path("./outputs/threshold_exploration")

BENCHMARKS = [1, 2, 3, 4, 5]
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
MAX_ALLOWED_MISSED_RATE = 0.05
EPS = 1e-7

# Candidate ranges are low/high P(hotspot) thresholds.
THRESHOLD_RANGES = [
    (0.10, 0.90),
    (0.15, 0.85),
    (0.20, 0.80),
    (0.30, 0.70),
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
        raise ValueError(f"Could not identify hotspot class from {classes}; expected hs and nhs folders.")
    hs_idx = hs_matches[0]
    xs, ys_class = [], []
    for x, y in ds:
        xs.append((tf.cast(x, tf.float32) / 255.0).numpy())
        ys_class.extend(y.numpy().astype(int).ravel().tolist())
    y_class = np.asarray(ys_class, dtype=int)
    y_hotspot = (1 - y_class) if hs_idx == 0 else y_class
    return np.concatenate(xs), y_hotspot.astype(int), classes, hs_idx


def get_hotspot_logits(model, x, hs_idx):
    # The model's single logit corresponds to class index 1.
    raw = model.predict(x, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    return -raw if hs_idx == 0 else raw


def fit_temperature(logits, y_true):
    y = np.asarray(y_true, dtype=int)

    def objective(log_t):
        temperature = np.exp(log_t)
        probabilities = tf.math.sigmoid(logits / temperature).numpy()
        probabilities = np.clip(probabilities, EPS, 1 - EPS)
        return log_loss(y, probabilities, labels=[0, 1])

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def summarize(y_true, probabilities, low=None, high=None):
    """If low/high are omitted, calculate ordinary binary baseline at 0.5."""
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    total = len(y_true)
    hotspot_mask = y_true == 1
    n_hotspots = int(hotspot_mask.sum())

    if low is None:
        # Standard baseline: predict hotspot when P(hotspot) >= 0.5.
        predicted_hotspot = p >= 0.5
        fn = int(np.sum(hotspot_mask & ~predicted_hotspot))
        tn = int(np.sum((~hotspot_mask) & ~predicted_hotspot))
        fp = int(np.sum((~hotspot_mask) & predicted_hotspot))
        tp = int(np.sum(hotspot_mask & predicted_hotspot))
        return {
            "total_images": total, "actual_hotspots": n_hotspots,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "baseline_missed_hotspot_rate_pct": 100 * fn / n_hotspots if n_hotspots else np.nan,
            "baseline_false_negative_rate_pct": 100 * fn / n_hotspots if n_hotspots else np.nan,
            "baseline_accuracy_pct": 100 * (tp + tn) / total if total else np.nan,
        }

    # Referral system: low => safe, high => risky, otherwise needs_check.
    safe = p <= low
    risky = p >= high
    needs_check = ~(safe | risky)
    missed = int(np.sum(safe & hotspot_mask))
    return {
        "total_images": total,
        "actual_hotspots": n_hotspots,
        "safe_count": int(safe.sum()),
        "risky_count": int(risky.sum()),
        "needs_check_count": int(needs_check.sum()),
        "safe_rate_pct": 100 * safe.mean() if total else np.nan,
        "risky_rate_pct": 100 * risky.mean() if total else np.nan,
        "referral_rate_pct": 100 * needs_check.mean() if total else np.nan,
        "missed_hotspots": missed,
        "missed_hotspot_rate_pct": 100 * missed / n_hotspots if n_hotspots else np.nan,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_benchmark_rows = []
    candidate_rows = []
    baseline_rows = []

    for benchmark in BENCHMARKS:
        model_path = MODEL_DIR / f"model_benchmark_{benchmark}.keras"
        meta_path = MODEL_DIR / f"model_benchmark_{benchmark}_metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing model/metadata for ICCAD-{benchmark}.\n"
                f"Expected:\n  {model_path}\n  {meta_path}\n"
                "Run 01_train_models.py first, and ensure its OUTPUT_DIR equals MODEL_DIR in this script."
            )

        x_val, y_val, classes, hs_idx = load_validation(benchmark)
        model = tf.keras.models.load_model(model_path)
        logits = get_hotspot_logits(model, x_val, hs_idx)

        # Baseline is the original, uncalibrated model at P(hotspot) = 0.5.
        p_raw = tf.math.sigmoid(logits).numpy()
        base = summarize(y_val, p_raw)
        baseline_rows.append({"benchmark": benchmark, **base})

        # Fit temperature only on validation data, then explore thresholds.
        temperature = fit_temperature(logits, y_val)
        p_cal = tf.math.sigmoid(logits / temperature).numpy()
        per_benchmark_rows.append({
            "benchmark": benchmark,
            "temperature_T": temperature,
            "validation_images": len(y_val),
            "validation_hotspots": int(y_val.sum()),
            "baseline_missed_hotspot_rate_pct": base["baseline_missed_hotspot_rate_pct"],
        })

        for low, high in THRESHOLD_RANGES:
            stats = summarize(y_val, p_cal, low, high)
            candidate_rows.append({
                "benchmark": benchmark,
                "threshold_range": f"{int(low*100)}-{int(high*100)}",
                "low_threshold": low,
                "high_threshold": high,
                "temperature_T": temperature,
                **stats,
                "meets_5pct_constraint_this_benchmark":
                    bool(stats["missed_hotspot_rate_pct"] <= 100 * MAX_ALLOWED_MISSED_RATE),
            })

        pd.DataFrame({
            "true_label": y_val,
            "hotspot_logit": logits,
            "raw_hotspot_probability": p_raw,
            "calibrated_hotspot_probability": p_cal,
        }).to_csv(OUTPUT_DIR / f"validation_predictions_benchmark_{benchmark}.csv", index=False)

        print(f"ICCAD-{benchmark}: baseline missed={base['baseline_missed_hotspot_rate_pct']:.3f}% | T={temperature:.4f}")

    baseline_df = pd.DataFrame(baseline_rows)
    per_benchmark_df = pd.DataFrame(candidate_rows)
    temp_df = pd.DataFrame(per_benchmark_rows)

    # Pooled metrics are calculated from counts across all benchmarks, not
    # by averaging the benchmark percentages.
    pooled_rows = []
    for (label, low, high), group in per_benchmark_df.groupby(
        ["threshold_range", "low_threshold", "high_threshold"], sort=False
    ):
        total_images = int(group["total_images"].sum())
        total_hotspots = int(group["actual_hotspots"].sum())
        total_missed = int(group["missed_hotspots"].sum())
        total_referrals = int(group["needs_check_count"].sum())
        worst_benchmark_miss = float(group["missed_hotspot_rate_pct"].max())
        pooled_rows.append({
            "threshold_range": label,
            "low_threshold": low,
            "high_threshold": high,
            "pooled_images": total_images,
            "pooled_actual_hotspots": total_hotspots,
            "pooled_missed_hotspots": total_missed,
            "pooled_missed_hotspot_rate_pct": 100 * total_missed / total_hotspots if total_hotspots else np.nan,
            "pooled_referral_rate_pct": 100 * total_referrals / total_images if total_images else np.nan,
            "worst_benchmark_missed_hotspot_rate_pct": worst_benchmark_miss,
            "meets_5pct_worst_benchmark_constraint":
                bool(worst_benchmark_miss <= 100 * MAX_ALLOWED_MISSED_RATE),
        })

    pooled_df = pd.DataFrame(pooled_rows)
    baseline_df.to_csv(OUTPUT_DIR / "baseline_validation_metrics_by_benchmark.csv", index=False)
    temp_df.to_csv(OUTPUT_DIR / "validation_temperature_summary.csv", index=False)
    per_benchmark_df.to_csv(OUTPUT_DIR / "threshold_metrics_per_benchmark.csv", index=False)
    pooled_df.to_csv(OUTPUT_DIR / "threshold_metrics_pooled_summary.csv", index=False)

    print("\nBASELINE MISSED-HOTSPOT RATE (raw probability threshold = 0.5)")
    print(baseline_df[["benchmark", "actual_hotspots", "FN",
                       "baseline_missed_hotspot_rate_pct"]].to_string(index=False))

    print("\nTHRESHOLD RANGE SUMMARY (validation; temperature-scaled probabilities)")
    print(pooled_df[["threshold_range", "pooled_missed_hotspot_rate_pct",
                     "worst_benchmark_missed_hotspot_rate_pct",
                     "pooled_referral_rate_pct",
                     "meets_5pct_worst_benchmark_constraint"]].to_string(index=False))

    print("\nSaved CSV files in:", OUTPUT_DIR.resolve())
    print("No test images were loaded. No candidate was automatically selected.")


if __name__ == "__main__":
    main()
