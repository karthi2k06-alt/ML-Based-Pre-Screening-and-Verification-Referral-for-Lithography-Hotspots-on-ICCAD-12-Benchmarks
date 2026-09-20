"""
Group 19 - ML-Based Pre-Screening and Verification Referral for Hotspots
STAGE 1 (Sai Kiran): Data loading + baseline model training

CORRECTED VERSION - see the one change flagged "FIX" below.

WHAT THIS SCRIPT DOES, IN PLAIN WORDS
--------------------------------------
1. Loads one ICCAD-12 benchmark's layout pictures (hotspot / non-hotspot folders).
2. Trains a small CNN to guess hotspot vs. non-hotspot.
3. Saves the trained model.
4. Runs the model on the test set and saves, for every test picture, the
   CONFIDENCE SCORE (a number 0-1) the model gave it - not just yes/no.
   Kamal's script (stage 2) needs this confidence-score file to build the
   "trust it / send for double-check" logic.

FOLDER STRUCTURE THIS SCRIPT EXPECTS
--------------------------------------
Matches the official "iccad-official" download:

    DATA_ROOT/
      iccad1/
        train/
          hs/       *.png or *.jpg
          nhs/      *.png or *.jpg
        test/
          hs/       *.png or *.jpg
          nhs/      *.png or *.jpg
      iccad2/
        ... same structure ...
      iccad3/  iccad4/  iccad5/

HOW TO RUN
--------------------------------------
1. pip install tensorflow scikit-learn pandas   (--break-system-packages if needed)
2. Edit DATA_ROOT below if your extracted "iccad-official" folder lives somewhere else.
3. python 01_baseline_model.py
4. Look in outputs/ for:
     - model_benchmark_<n>.keras         (trained model, one per benchmark)
     - predictions_benchmark_<n>.csv     (filename, true label, confidence score)
     - baseline_metrics_summary.csv      (accuracy/precision/recall/etc per benchmark)

WHAT WAS WRONG AND WHY (read this before you re-run)
--------------------------------------
The original script did:

    summary.loc["average"] = summary.mean(numeric_only=True)

This sets the pandas INDEX label to the string "average", but the
"benchmark" COLUMN for that row becomes the numeric mean of 1,2,3,4,5,
which is exactly 3.0 - indistinguishable, once written to CSV, from the
real Benchmark 3 row. Downstream, 03_final_comparison.py filters with
`baseline["benchmark"].isin([1,2,3,4,5])`, which does NOT catch this,
because 3.0 legitimately passes that filter. The result: Benchmark 3
appears TWICE in baseline_metrics_summary.csv (the real row, and this
disguised average row), and when 03_final_comparison.py merges it against
referral_summary.csv, every Benchmark-3 threshold row gets matched
against BOTH baseline rows - producing 6 rows instead of 3 for Benchmark 3
in comparison_table.csv, half of them wrong.

THE FIX: make the average row's "benchmark" value the STRING "average"
instead of a number, so it can never collide with a real benchmark number,
and so any numeric filter on "benchmark" naturally excludes it.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# CONFIG - edit these two lines for your setup
# ---------------------------------------------------------------------------
DATA_ROOT = "./iccad-official"  # <-- point this at the extracted iccad-official folder
OUTPUT_DIR = "./outputs"
BENCHMARKS = [1, 2, 3, 4, 5]  # iccad1 .. iccad5
IMG_SIZE = (64, 64)           # small size keeps training fast; bump up if you have time
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.15       # 15% of each benchmark's TRAIN folder held out for validation

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1: Load a benchmark's train/val/test images from disk
# ---------------------------------------------------------------------------
def load_train_and_val(benchmark_num):
    """
    Loads the 'train' folder of one benchmark, then splits it into a
    TRAINING portion and a held-out VALIDATION portion (15% of train,
    never seen during weight updates - used only to sanity-check the
    model while it trains). The TEST set is completely separate and is
    still only touched once, at the end, in load_test_with_paths().
    """
    folder = os.path.join(DATA_ROOT, f"iccad{benchmark_num}", "train")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=VALIDATION_SPLIT,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,                       # same seed as above - required so the
        validation_split=VALIDATION_SPLIT,  # train/val split is the complementary
        subset="validation",           # half, with no overlap and no leakage
    )

    class_names = train_ds.class_names   # e.g. ['hs', 'nhs'] - check the order!
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
    return train_ds, val_ds, class_names


def load_test_with_paths(benchmark_num):
    """
    Same as above but keeps file paths aligned to predictions, unshuffled.
    Needed so Kamal's script can match a confidence score back to a specific image.
    """
    folder = os.path.join(DATA_ROOT, f"iccad{benchmark_num}", "test")
    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    file_paths = dataset.file_paths
    class_names = dataset.class_names
    dataset = dataset.map(lambda x, y: (x / 255.0, y))
    return dataset, file_paths, class_names


def find_hotspot_index(class_names):
    """
    Figures out which class name means 'hotspot', whether it's called
    'hs', 'train_hs', 'test_hs', etc. The non-hotspot folder's name
    always contains the letters 'nhs' somewhere (nonhotspot, train_nhs,
    test_nhs, nhs...) - the hotspot one never does.
    """
    for i, name in enumerate(class_names):
        if "nhs" not in name.lower():
            return i
    raise ValueError(f"Could not identify the hotspot class among: {class_names}")


# ---------------------------------------------------------------------------
# STEP 2: Define a small CNN (own implementation - not copied from any repo)
# ---------------------------------------------------------------------------
def build_model():
    """
    A small CNN: two conv+pool blocks, then a dense layer, ending in a single
    sigmoid output (a number between 0 and 1 = the model's confidence that
    the picture is a hotspot).
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),   # <-- this output IS the confidence score
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# STEP 3: Train + evaluate one benchmark
# ---------------------------------------------------------------------------
def run_one_benchmark(benchmark_num, class_weight_boost=True):
    print(f"\n=== Benchmark {benchmark_num} ===")

    train_ds, val_ds, class_names = load_train_and_val(benchmark_num)
    test_ds, test_paths, test_class_names = load_test_with_paths(benchmark_num)
    if class_names != test_class_names:
        print(f"Note: train classes {class_names} vs test classes {test_class_names} "
              f"- order differs, handling each independently, no action needed.")

    # hotspot_index tells us which output (0 or 1) means hotspot - this can
    # differ between train and test, so we compute each independently.
    hotspot_index = find_hotspot_index(class_names)
    test_hotspot_index = find_hotspot_index(test_class_names)

    # Because hotspots are rare, tell the model to pay more attention to them
    # during training (this does NOT change your test set - only training weighting).
    class_weight = None
    if class_weight_boost:
        # quick pass to count how many of each class are in the training set
        n_hotspot, n_nonhotspot = 0, 0
        for _, y in train_ds.unbatch():
            if int(y.numpy()[0]) == hotspot_index:
                n_hotspot += 1
            else:
                n_nonhotspot += 1
        total = n_hotspot + n_nonhotspot
        class_weight = {
            hotspot_index: total / (2 * max(n_hotspot, 1)),
            1 - hotspot_index: total / (2 * max(n_nonhotspot, 1)),
        }
        print(f"Class counts -> hotspot: {n_hotspot}, non-hotspot: {n_nonhotspot}")
        print(f"Using class_weight: {class_weight}")

    model = build_model()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        verbose=1,
    )
    val_accuracy = history.history.get("val_accuracy", [None])[-1]
    print(f"Final validation accuracy (held-out {VALIDATION_SPLIT*100:.0f}% of train): {val_accuracy}")

    model.save(os.path.join(OUTPUT_DIR, f"model_benchmark_{benchmark_num}.keras"))

    # --- Get predictions + confidence scores on the test set ---
    y_true, y_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).flatten()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # If hotspot_index == 0, a HIGH sigmoid output actually means "non-hotspot" -
    # flip the score so "high score = more likely hotspot" consistently, which
    # is what Kamal's thresholding logic will expect. Predictions are flipped
    # based on the TRAINING label order (that's what the model actually learned);
    # true labels are flipped based on the TEST label order (that's how the test
    # folders were actually read).
    y_prob_hotspot = (1 - y_prob) if hotspot_index == 0 else y_prob
    y_true_hotspot = (1 - y_true) if test_hotspot_index == 0 else y_true

    results_df = pd.DataFrame({
        "file_path": test_paths,
        "true_label": y_true_hotspot.astype(int),      # 1 = hotspot, 0 = non-hotspot
        "hotspot_confidence": y_prob_hotspot,           # 0-1, higher = more likely hotspot
    })
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, f"predictions_benchmark_{benchmark_num}.csv"),
        index=False,
    )

    # --- Basic metrics at the standard 0.5 cutoff, just to sanity-check the model ---
    y_pred = (y_prob_hotspot >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_hotspot, y_pred).ravel()
    precision = precision_score(y_true_hotspot, y_pred, zero_division=0)
    recall = recall_score(y_true_hotspot, y_pred, zero_division=0)          # = sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true_hotspot, y_pred, zero_division=0)
    balanced_acc = (recall + specificity) / 2

    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f} Specificity={specificity:.3f} "
          f"F1={f1:.3f} BalancedAcc={balanced_acc:.3f}")

    return {
        "benchmark": benchmark_num,
        "val_accuracy": val_accuracy,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "balanced_accuracy": balanced_acc,
    }


# ---------------------------------------------------------------------------
# MAIN: run all 5 benchmarks, save a summary table
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_results = []
    for b in BENCHMARKS:
        all_results.append(run_one_benchmark(b))

    summary = pd.DataFrame(all_results)

    # FIX: the average row's "benchmark" value must be a STRING, never a number,
    # so it can never be mistaken for (or accidentally match) a real benchmark
    # id downstream. Using pd.concat instead of the old .loc["average"]=... also
    # avoids silently coercing the whole "benchmark" column's dtype in a way
    # that depends on pandas version.
    avg_row = summary.mean(numeric_only=True).to_dict()
    avg_row["benchmark"] = "average"
    summary_with_avg = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)

    summary_with_avg.to_csv(os.path.join(OUTPUT_DIR, "baseline_metrics_summary.csv"), index=False)
    print("\nDone. See outputs/baseline_metrics_summary.csv for the full comparison table.")
    print("(Per-benchmark rows have a numeric 'benchmark' column; the overall-average "
          "row is explicitly labeled benchmark='average' so it can never be confused "
          "with Benchmark 3 or any other real benchmark again.)")
