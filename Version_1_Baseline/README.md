# Version 1 – Baseline CNN

## Overview
Version 1 implements the baseline CNN model for lithography hotspot detection using the five ICCAD benchmarks.

## Methodology
- Uses a Convolutional Neural Network (CNN).
- Performs binary classification.
- Classifies layout images as Hotspot or Non-Hotspot.
- Uses a fixed probability threshold of 0.5.
- Evaluates performance using test data.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Missed Hotspot Rate
- Confusion Matrix

## Purpose
This version serves as the baseline for comparison with Version 2, which introduces a calibrated CNN-based referral approach.
