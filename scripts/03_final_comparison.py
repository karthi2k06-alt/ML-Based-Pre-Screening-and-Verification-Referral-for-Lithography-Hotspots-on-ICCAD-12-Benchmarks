"""
Group 19 - ML-Based Pre-Screening and Verification Referral for Hotspots
STAGE 3 (Karthikeyan): Final comparison, tables and charts

UPDATED VERSION. Two kinds of changes:

  1. BUG FIX: the benchmark filter now works even if baseline_metrics_summary.csv
     still has the old numeric "3.0"-labeled average row in it (belt-and-braces:
     even though 01_baseline_model.py has been fixed to label it "average",
     this script no longer silently trusts a numeric-looking value to mean a
     real benchmark). It explicitly keeps only rows whose benchmark is one of
     1,2,3,4,5 AFTER coercing to numeric, and reports how many rows were
     dropped so a bad merge is never silent again.

  2. NEW OUTPUTS the assignment/rubric asks for that weren't being produced:
     - screening_efficiency_table.csv (screening/flagged/referral rate per
       benchmark per threshold, standalone)
     - proposed_bucket_confusion.csv (true HS/NHS vs safe/risky/needs_check
       counts - the "confusion matrix" for the proposed system, which is a
       2x3 table since it's a 3-way decision, not a 2x2)
     - overall_average_summary.csv (mean across the 5 benchmarks, for both
       baseline and proposed, per threshold - required by the rubric's
       "report results per benchmark AND the overall average")
     - chart_screening_efficiency.png (screening rate vs threshold, per
       benchmark - shows the workload-reduction side of the trade-off, which
       the original two charts didn't show on their own)

WHAT THIS SCRIPT DOES, IN PLAIN WORDS
--------------------------------------
Sai Kiran gave us:  baseline_metrics_summary.csv
Kamal gave us:      referral_summary.csv
And Kamal's per-picture bucket files: referral_labels_benchmark_*_thresh_*.csv

This script puts them side by side to answer the actual research question:
"Does adding the double-check (referral) system actually catch more real
hotspots than the plain baseline model did on its own, and how much workload
does pre-screening remove?"

It produces:
  1. comparison_table.csv            - baseline vs. proposed missed-rate, per
                                        benchmark per threshold
  2. screening_efficiency_table.csv  - screening/flagged/referral rate, per
                                        benchmark per threshold
  3. proposed_bucket_confusion.csv   - true label x bucket counts (2x3), per
                                        benchmark per threshold
  4. overall_average_summary.csv     - mean across the 5 benchmarks
  5. chart_missed_hotspots.png       - bar chart: baseline vs proposed, per benchmark
  6. chart_tradeoff.png              - line chart: referral rate vs missed rate
  7. chart_screening_efficiency.png  - line chart: screening rate vs threshold

HOW TO RUN
--------------------------------------
1. pip install matplotlib   (if not already installed - "py -3.12 -m pip install matplotlib")
2. Put this script in the same folder as your other two scripts (Desktop),
   with the outputs/ folder containing baseline_metrics_summary.csv,
   referral_summary.csv, and the referral_labels_*.csv files already in it.
3. py -3.12 03_final_comparison.py
4. Look in outputs/ for the files listed above.
"""

import glob
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "./outputs"

BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline_metrics_summary.csv")
REFERRAL_PATH = os.path.join(OUTPUT_DIR, "referral_summary.csv")

# Which threshold setting to feature in the main bar chart (there are 3 in
# referral_summary.csv - we use the "medium" one as the headline comparison,
# and show all 3 in the trade-off chart).
FEATURED_LOW_THRESHOLD = 0.20
FEATURED_HIGH_THRESHOLD = 0.80


def load_baseline():
    baseline = pd.read_csv(BASELINE_PATH)

    # FIX: coerce to numeric first, THEN filter. This is robust regardless of
    # whether the "average" row was written as the string "average" (current,
    # fixed 01_baseline_model.py) or, if anyone regenerates this file with an
    # older/unfixed version, as a bare 3.0 that collides with Benchmark 3 -
    # in the latter case we can't recover the correct row automatically, so
    # we fail loudly instead of silently duplicating rows.
    n_rows_before = len(baseline)
    numeric_benchmark = pd.to_numeric(baseline["benchmark"], errors="coerce")
    baseline = baseline[numeric_benchmark.isin([1, 2, 3, 4, 5])].copy()
    baseline["benchmark"] = numeric_benchmark[numeric_benchmark.isin([1, 2, 3, 4, 5])].astype(int)
    n_rows_after = len(baseline)

    if n_rows_after != 5:
        raise ValueError(
            f"Expected exactly 5 per-benchmark rows in baseline_metrics_summary.csv "
            f"after filtering, got {n_rows_after} (started with {n_rows_before} total "
            f"rows). This usually means the 'average' row is colliding with a real "
            f"benchmark id again - re-run the FIXED 01_baseline_model.py to regenerate "
            f"baseline_metrics_summary.csv, don't patch this file by hand."
        )

    baseline["baseline_missed_hotspot_rate"] = 1 - baseline["recall_sensitivity"]
    return baseline


def load_and_merge():
    baseline = load_baseline()
    referral = pd.read_csv(REFERRAL_PATH)

    merged = referral.merge(
        baseline[["benchmark", "baseline_missed_hotspot_rate", "balanced_accuracy"]]
        .rename(columns={"balanced_accuracy": "baseline_balanced_accuracy"}),
        on="benchmark",
        how="left",
    )

    # How much better (or worse) is the proposed referral system, in
    # percentage points of missed-hotspot-rate?
    merged["improvement_pp"] = (
        (merged["baseline_missed_hotspot_rate"] - merged["missed_hotspot_rate"]) * 100
    )

    return merged, baseline


def save_comparison_table(merged):
    cols = [
        "benchmark", "low_threshold", "high_threshold",
        "baseline_missed_hotspot_rate", "missed_hotspot_rate",
        "improvement_pp", "referral_rate",
    ]
    table = merged[cols].round(4)
    table.to_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"), index=False)
    print("Saved comparison_table.csv\n")
    print(table.to_string(index=False))
    return table


def save_screening_efficiency_table(merged):
    """
    NEW. Standalone table of screening/flagged/referral rate, so this isn't
    only implicit inside referral_summary.csv. screening_rate and
    flagged_rate columns exist only if 02_referral_logic.py has been
    re-run with the updated version; if they're missing (old
    referral_summary.csv), we fall back to leaving them out rather than
    inventing numbers.
    """
    cols = ["benchmark", "low_threshold", "high_threshold", "referral_rate"]
    optional_cols = ["screening_rate", "flagged_rate"]
    for c in optional_cols:
        if c in merged.columns:
            cols.insert(-1, c)
        else:
            print(f"Note: '{c}' not found in referral_summary.csv - re-run the "
                  f"updated 02_referral_logic.py to include it. Skipping for now.")

    table = merged[cols].round(4)
    table.to_csv(os.path.join(OUTPUT_DIR, "screening_efficiency_table.csv"), index=False)
    print("\nSaved screening_efficiency_table.csv")
    return table


def build_bucket_confusion_table():
    """
    NEW. Reads every referral_labels_benchmark_<n>_thresh_<low>_<high>.csv
    file and builds a 2x3 "confusion matrix" (true HS / true NHS, each split
    across safe / needs_check / risky) per benchmark per threshold setting.
    This is the proposed system's equivalent of a confusion matrix - it's
    2x3, not 2x2, because the proposed system makes a 3-way decision.
    """
    pattern = os.path.join(OUTPUT_DIR, "referral_labels_benchmark_*_thresh_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("\nNo referral_labels_*.csv files found - skipping bucket confusion table.")
        return None

    name_re = re.compile(
        r"referral_labels_benchmark_(\d+)_thresh_([0-9.]+)_([0-9.]+)\.csv$"
    )

    rows = []
    for path in files:
        m = name_re.search(os.path.basename(path))
        if not m:
            continue
        benchmark_num, low_threshold, high_threshold = m.groups()
        df = pd.read_csv(path)

        for true_label, true_name in [(1, "true_HS"), (0, "true_NHS")]:
            subset = df[df["true_label"] == true_label]
            rows.append({
                "benchmark": int(benchmark_num),
                "low_threshold": float(low_threshold),
                "high_threshold": float(high_threshold),
                "true_class": true_name,
                "n_total": len(subset),
                "n_safe": (subset["bucket"] == "safe").sum(),
                "n_needs_check": (subset["bucket"] == "needs_check").sum(),
                "n_risky": (subset["bucket"] == "risky").sum(),
            })

    table = pd.DataFrame(rows).sort_values(
        ["benchmark", "low_threshold", "true_class"]
    )
    table.to_csv(os.path.join(OUTPUT_DIR, "proposed_bucket_confusion.csv"), index=False)
    print("Saved proposed_bucket_confusion.csv")
    return table


def save_overall_average_summary(merged, baseline):
    """
    NEW. Mean across the 5 benchmarks, for both baseline and proposed, per
    threshold setting - the rubric explicitly asks for per-benchmark results
    "along with the overall average".
    """
    baseline_avg = baseline[
        ["val_accuracy", "precision", "recall_sensitivity", "specificity",
         "f1_score", "balanced_accuracy", "baseline_missed_hotspot_rate"]
    ].mean(numeric_only=True)
    baseline_avg_df = pd.DataFrame([baseline_avg])
    baseline_avg_df.insert(0, "scope", "baseline_overall_average")
    baseline_avg_df.round(4).to_csv(
        os.path.join(OUTPUT_DIR, "overall_average_summary.csv"), index=False
    )

    proposed_cols = ["low_threshold", "high_threshold", "referral_rate",
                      "missed_hotspot_rate", "improvement_pp"]
    for c in ["screening_rate", "flagged_rate"]:
        if c in merged.columns:
            proposed_cols.insert(-2, c)

    proposed_avg = (
        merged.groupby(["low_threshold", "high_threshold"])[
            [c for c in proposed_cols if c not in ("low_threshold", "high_threshold")]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    proposed_avg.insert(0, "scope", "proposed_overall_average")

    combined = pd.concat([baseline_avg_df, proposed_avg], ignore_index=True)
    combined.round(4).to_csv(
        os.path.join(OUTPUT_DIR, "overall_average_summary.csv"), index=False
    )
    print("Saved overall_average_summary.csv")
    return combined


def chart_missed_hotspots(merged):
    featured = merged[
        (merged["low_threshold"] == FEATURED_LOW_THRESHOLD)
        & (merged["high_threshold"] == FEATURED_HIGH_THRESHOLD)
    ].sort_values("benchmark")

    benchmarks = featured["benchmark"].astype(str)
    x = range(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], featured["baseline_missed_hotspot_rate"] * 100,
           width, label="Baseline (no referral)")
    ax.bar([i + width / 2 for i in x], featured["missed_hotspot_rate"] * 100,
           width, label=f"Proposed (referral, thresholds {FEATURED_LOW_THRESHOLD}/{FEATURED_HIGH_THRESHOLD})")

    ax.set_xlabel("ICCAD-12 Benchmark")
    ax.set_ylabel("Missed Hotspot Rate (%)")
    ax.set_title("Baseline vs. Proposed: Real Hotspots Missed, per Benchmark")
    ax.set_xticks(list(x))
    ax.set_xticklabels(benchmarks)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "chart_missed_hotspots.png"), dpi=150)
    plt.close(fig)
    print("\nSaved chart_missed_hotspots.png")


def chart_tradeoff(merged):
    fig, ax = plt.subplots(figsize=(8, 5))
    for benchmark_num, group in merged.groupby("benchmark"):
        group = group.sort_values("referral_rate")
        ax.plot(group["referral_rate"] * 100, group["missed_hotspot_rate"] * 100,
                marker="o", label=f"Benchmark {benchmark_num}")

    ax.set_xlabel("Referral Rate (% of pictures sent for double-check)")
    ax.set_ylabel("Missed Hotspot Rate (%)")
    ax.set_title("Trade-off: Referring More Pictures vs. Missing Fewer Hotspots")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "chart_tradeoff.png"), dpi=150)
    plt.close(fig)
    print("Saved chart_tradeoff.png")


def chart_screening_efficiency(merged):
    """NEW. Shows the workload-reduction side of the trade-off directly."""
    if "screening_rate" not in merged.columns:
        print("\n'screening_rate' not in referral_summary.csv - re-run the updated "
              "02_referral_logic.py to get this chart. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for benchmark_num, group in merged.groupby("benchmark"):
        group = group.sort_values("referral_rate")
        ax.plot(group["referral_rate"] * 100, group["screening_rate"] * 100,
                marker="o", label=f"Benchmark {benchmark_num}")

    ax.set_xlabel("Referral Rate (% sent for double-check)")
    ax.set_ylabel("Screening Rate (% screened out with no check at all)")
    ax.set_title("Trade-off: Referral Rate vs. Screening Efficiency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "chart_screening_efficiency.png"), dpi=150)
    plt.close(fig)
    print("Saved chart_screening_efficiency.png")


if __name__ == "__main__":
    merged, baseline = load_and_merge()
    save_comparison_table(merged)
    save_screening_efficiency_table(merged)
    build_bucket_confusion_table()
    save_overall_average_summary(merged, baseline)
    chart_missed_hotspots(merged)
    chart_tradeoff(merged)
    chart_screening_efficiency(merged)
    print("\nDone. Tables + charts are in outputs/ - ready for the report.")
