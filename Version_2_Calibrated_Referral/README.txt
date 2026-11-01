GROUP 19 — CNN HOTSPOT PRE-SCREENING PIPELINE

FLOW
1. Put the dataset in ./iccad-official/iccad1 ... iccad5. Each benchmark must contain:
   train/hs, train/nhs, validation/hs, validation/nhs, test/hs, test/nhs
   Folder names may be hs/nhs or names containing those tokens; non-hotspot folder name must contain 'nhs'.
2. Edit DATA_ROOT in each script if your dataset root differs.
3. Install dependencies in the Python environment used for TensorFlow:
   python -m pip install tensorflow pandas numpy scikit-learn scipy matplotlib
4. Run in order:
   python 01_train_models.py
   python 02_calibrate_and_select_thresholds.py
   python 03_evaluate_test_and_plot.py

IMPORTANT METHODOLOGY NOTES
- Training uses train only for weight updates and validation for early stopping. Test is not read until stage 3.
- CNN architecture retains 64x64 grayscale input, Conv2D(16,3x3), BN, MaxPool, Conv2D(32,3x3), BN, MaxPool, Flatten, Dropout(0.3), Dense(64). Final Dense emits a logit instead of sigmoid so temperature scaling can use z/T. For baseline 0.5 decisions, sigmoid(logit)>=0.5 is equivalent to logit>=0.
- Temperature T is fitted separately per benchmark by minimizing validation binary cross-entropy/NLL. Calibration uses the model's hotspot logit and validation labels only.
- Referral buckets: calibrated P(hotspot)<=low => safe; >=high => risky; otherwise needs_check.
- The 5% criterion is treated as a MAXIMUM allowed missed-hotspot rate (<=5%), not a minimum. Selection checks the worst per-benchmark validation rate. Among eligible candidates, choose the lowest pooled referral rate; tie-break on lower worst-case miss rate.
- Candidate pairs are 0.10/0.90, 0.20/0.80, 0.30/0.70. Change THRESHOLD_RANGES only before running stage 2, and document why.
- Test data is used only after T and thresholds have been frozen. Do not change thresholds based on test results.
- Proposed missed-hotspot rate means true hotspots classified SAFE without referral / all true hotspots. A referred hotspot is not counted as missed because it is sent for further verification. This is a workload proxy, not proof of actual downstream verification savings.
- Proposed confusion matrix is an operational binary view: SAFE is negative/pass; RISKY or NEEDS_CHECK is positive/not passed as safe. NEEDS_CHECK cases do not have verified downstream labels, so this matrix is not a final post-human-review confusion matrix. Keep the separate 3-bucket metrics in the report.
- If no candidate meets the <=5% constraint on validation, the script stops. Do not silently relax the constraint or tune on test; add candidate ranges or improve the model using validation only.

MAIN OUTPUTS (./outputs)
- model_benchmark_<n>.keras and metadata JSON
- temperature_scaling_summary.csv
- validation_calibrated_predictions_benchmark_<n>.csv
- validation_threshold_metrics_per_benchmark.csv
- validation_threshold_selection.csv
- final_calibration_and_thresholds.json
- test_baseline_metrics.csv and test_proposed_referral_metrics.csv
- test_baseline_vs_proposed_comparison.csv and test_predictions_benchmark_<n>.csv
- confusion_matrix_baseline_iccad<n>.png, confusion_matrix_proposed_iccad<n>.png, pooled confusion matrices
- chart_missed_hotspots_baseline_vs_proposed.png
- chart_validation_missed_vs_referral.png
