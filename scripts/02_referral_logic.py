"""
Group 19 - ML-Based Pre-Screening and Verification Referral for Hotspots
STAGE 2 (Kamal): The referral / double-check logic

WHAT THIS SCRIPT DOES, IN PLAIN WORDS
--------------------------------------
Sai Kiran's script already gave every test picture a confidence score
(0.0 to 1.0 - how sure the model is that it's a hotspot).

This script takes those scores and sorts every picture into one of 3 buckets:

  - "safe"        -> model is very confident it's NOT a hotspot -> let it pass
  - "risky"       -> model is very confident it IS a hotspot   -> flag it
  - "needs_check" -> model isn't confident either way          -> send for
                                                                   human/simulation
                                                                   double-check

This is your team's actual new idea (the "pre-screening and verification
referral" part of your project title). Everything before this was just the
plain baseline.

We also try a few DIFFERENT threshold settings (how strict "confident"
means) and compare them - this doubles as your required ablation study.

HOW TO RUN
--------------------------------------
1. Make sure the outputs/ folder from Sai Kiran's script (with
   predictions_benchmark_1.csv ... predictions_benchmark_5.csv) is in the
   same folder as this script, OR edit PREDICTIONS_DIR below.
2. python 02_referral_logic.py   (or  py -3.12 02_referral_logic.py)
3. Look in outputs/ for:
     - referral_labels_benchmark_<n>_thresh_<low>_<high>.csv
         (every test picture's bucket, for each threshold setting tried)
     - referral_summary.csv
         (one row per benchmark per threshold setting, with the key numbers)
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PREDICTIONS_DIR = "./outputs"   # where Sai Kiran's predictions_benchmark_*.csv files are
OUTPUT_DIR = "./outputs"        # where this script writes its own results
BENCHMARKS = [1, 2, 3, 4, 5]

# Each pair below is (low_threshold, high_threshold).
# Meaning for a picture with confidence score C:
#   C <= low_threshold   -> "safe"          (confident it's NOT a hotspot)
#   C >= high_threshold  -> "risky"         (confident it IS a hotspot)
#   otherwise            -> "needs_check"   (not confident, send for double-check)
#
# We try 3 settings, from loose to strict, so you can compare them
# (this is your ablation study - just showing 3 different settings and how
# the results change).
THRESHOLD_SETTINGS = [
    (0.30, 0.70),   # loose  - wide "needs_check" zone, safest but more referrals
    (0.20, 0.80),   # medium
    (0.10, 0.90),   # strict - narrow "needs_check" zone, fewer referrals, riskier
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1: Turn a confidence score into a safe / risky / needs_check label
# ---------------------------------------------------------------------------
def bucket_label(confidence, low_threshold, high_threshold):
    if confidence <= low_threshold:
        return "safe"
    elif confidence >= high_threshold:
        return "risky"
    else:
        return "needs_check"


# ---------------------------------------------------------------------------
# STEP 2: Apply this to one benchmark, for one threshold setting
# ---------------------------------------------------------------------------
def apply_referral(benchmark_num, low_threshold, high_threshold):
    path = os.path.join(PREDICTIONS_DIR, f"predictions_benchmark_{benchmark_num}.csv")
    df = pd.read_csv(path)

    df["bucket"] = df["hotspot_confidence"].apply(
        lambda c: bucket_label(c, low_threshold, high_threshold)
    )

    # Save the full per-picture result (useful for Karthikeyan's plots later)
    out_name = f"referral_labels_benchmark_{benchmark_num}_thresh_{low_threshold}_{high_threshold}.csv"
    df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)

    total = len(df)
    n_safe = (df["bucket"] == "safe").sum()
    n_risky = (df["bucket"] == "risky").sum()
    n_needs_check = (df["bucket"] == "needs_check").sum()

    # The most important number: how many REAL hotspots did we let through
    # as "safe" without any double-check? (this is the dangerous mistake)
    real_hotspots = df[df["true_label"] == 1]
    missed_hotspots = (real_hotspots["bucket"] == "safe").sum()
    total_real_hotspots = len(real_hotspots)
    missed_hotspot_rate = missed_hotspots / total_real_hotspots if total_real_hotspots > 0 else 0.0

    referral_rate = n_needs_check / total if total > 0 else 0.0

    return {
        "benchmark": benchmark_num,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "total_pictures": total,
        "n_safe": n_safe,
        "n_risky": n_risky,
        "n_needs_check": n_needs_check,
        "referral_rate": referral_rate,            # % of pictures sent for double-check
        "total_real_hotspots": total_real_hotspots,
        "missed_hotspots_marked_safe": missed_hotspots,  # dangerous mistake count
        "missed_hotspot_rate": missed_hotspot_rate,      # % of real hotspots let through unchecked
    }


# ---------------------------------------------------------------------------
# MAIN: run every benchmark x every threshold setting, save one summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_rows = []
    for benchmark_num in BENCHMARKS:
        for low_threshold, high_threshold in THRESHOLD_SETTINGS:
            print(f"Benchmark {benchmark_num}, thresholds ({low_threshold}, {high_threshold})...")
            row = apply_referral(benchmark_num, low_threshold, high_threshold)
            all_rows.append(row)
            print(
                f"  referral_rate={row['referral_rate']:.3f}  "
                f"missed_hotspot_rate={row['missed_hotspot_rate']:.3f}  "
                f"(missed {row['missed_hotspots_marked_safe']} of {row['total_real_hotspots']})"
            )

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "referral_summary.csv"), index=False)
    print("\nDone. See outputs/referral_summary.csv for the full comparison table.")
