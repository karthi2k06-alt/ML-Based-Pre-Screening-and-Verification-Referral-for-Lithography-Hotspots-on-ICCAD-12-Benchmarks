 ML-Based Pre-Screening and Verification Referral for Lithography Hotspots

## Project Overview

This project uses Machine Learning to detect lithography hotspots from the ICCAD-12 dataset.

The project has three stages:

1. Train a CNN model to detect hotspots.
2. Use the CNN confidence score to classify patterns as SAFE, NEEDS_CHECK, or RISKY.
3. Compare the proposed method with the normal binary classification method.

## Project Flow

```text
ICCAD-12 Images
       |
       v
   CNN Model
       |
       v
Confidence Score
       |
       +----------------------+
       |          |           |
      SAFE    NEEDS_CHECK    RISKY
       |          |           |
    Screen     Verify       Flag
     Out       Further
```
## Dataset
```text

The project uses the ICCAD-12 lithography hotspot dataset.

It contains 5 benchmarks:

Benchmark 1
Benchmark 2
Benchmark 3
Benchmark 4
Benchmark 5

The dataset contains hotspot (HS) and non-hotspot (NHS) layout patterns.
```
## CNN Model
```text

The CNN takes a grayscale layout image as input.

Input size:

64 x 64 x 1

The model contains:

Convolution layers
Batch Normalization
Max Pooling
Dropout
Dense layer
Sigmoid output

The output is a hotspot confidence score between 0 and 1.
```
## Baseline Method
```text

The baseline uses a threshold of 0.50.

Confidence >= 0.50  -> Hotspot

Confidence < 0.50   -> Non-Hotspot

The baseline is evaluated using:

Precision
Recall
Specificity
F1-score
Balanced Accuracy
```
## Proposed Method
```text
The proposed method uses two confidence thresholds.

Low confidence
      |
      v
    SAFE

Medium confidence
      |
      v
 NEEDS_CHECK

High confidence
      |
      v
    RISKY

The tested threshold pairs are:

0.30 / 0.70

0.20 / 0.80

0.10 / 0.90
```
## Main Result
```text
For the 0.20 / 0.80 threshold configuration:

Baseline unchecked hotspot rate : 11.94%

Proposed unchecked hotspot rate : 4.77%

Reduction                       : 7.17 percentage points

The proposed method sends uncertain patterns to the NEEDS_CHECK group instead of making a direct binary decision.
```
## Important Limitation
```text
The current project implements the referral system, but the actual detailed verification stage has not yet been performed.

Therefore:

SAFE        -> Pattern is screened out

NEEDS_CHECK -> Pattern requires further verification

RISKY       -> Pattern is flagged as likely hotspot

The reported missed-hotspot rate represents real hotspots that were routed to SAFE. It is not the final false-negative rate after actual verification.
```
## Project Files
```text
scripts/
    01_baseline_model.py
    02_referral_logic.py
    03_final_comparison.py

outputs/
    Prediction CSV files
    Referral results
    Comparison tables
    Charts

report/
    IEEE Report

reference/
    Reference Paper
```
## How to Run
```text
1. Prepare the Dataset

Place the ICCAD-12 dataset in:

./iccad-official/
2. Run the Baseline Model
python scripts/01_baseline_model.py
3. Run the Referral Logic
python scripts/02_referral_logic.py
4. Generate Final Results
python scripts/03_final_comparison.py
```
## Requirements
```text
The project uses:

Python
TensorFlow
NumPy
Pandas
Matplotlib
Scikit-learn
Pillow

Install the required packages using:

pip install numpy pandas matplotlib scikit-learn pillow tensorflow
```
## Future Work
```text
Future work includes:

Implementing the actual hotspot verification stage
Improving confidence calibration
Testing different CNN architectures
Testing additional datasets
Measuring actual verification time
Integrating the system into a complete lithography verification flow
