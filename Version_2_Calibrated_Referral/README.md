# Version 2 – Calibrated CNN-Based Referral Pipeline

## Overview
Version 2 extends the baseline CNN-based lithography hotspot detection pipeline by introducing probability calibration and validation-based threshold selection.

## Methodology
- Uses the trained CNN models from Version 1.
- Applies temperature scaling using validation data.
- Selects low and high probability thresholds based on validation performance.
- Uses three decision categories:
  - SAFE
  - NEEDS_CHECK
  - RISKY
- Evaluates the finalized referral pipeline using test data.

## Threshold Selection
Thresholds are selected using validation data while considering the maximum allowed missed-hotspot rate across the benchmarks.

## Evaluation Metrics
- Missed Hotspot Rate
- Referral Rate
- Screening Rate
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Purpose
This version aims to reduce missed hotspots while referring uncertain cases for additional verification.
