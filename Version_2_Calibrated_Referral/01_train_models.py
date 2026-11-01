"""Train one CNN per ICCAD benchmark using pre-split train/validation folders.
Expected: DATA_ROOT/iccad1/{train,validation,test}/{hs,nhs}/images
Test data is NOT loaded in this stage.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_ROOT = Path("../iccad_train_validation_split/iccad-official")
OUTPUT_DIR = Path("./outputs")
BENCHMARKS = [1, 2, 3, 4, 5]
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
USE_CLASS_WEIGHTS = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


def load_split(benchmark, split, shuffle=False):
    folder = DATA_ROOT / f"iccad{benchmark}" / split
    if not folder.exists():
        raise FileNotFoundError(f"Missing split folder: {folder}")
    ds = tf.keras.utils.image_dataset_from_directory(
        str(folder), labels="inferred", label_mode="binary", color_mode="grayscale",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=shuffle,
        seed=SEED if shuffle else None)
    class_names = ds.class_names
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def hotspot_index(class_names):
    matches = [i for i, name in enumerate(class_names) if "nhs" not in name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Could not unambiguously identify hs class from {class_names}. Use folder names containing 'hs' and 'nhs'.")
    return matches[0]


def build_model():
    # Same CNN feature architecture. Final layer emits a LOGIT (no sigmoid),
    # which is required for mathematically correct temperature scaling.
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling2D((2, 2)),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation=None, name="logit")
    ])
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0)])
    return model


def calculate_class_weights(train_ds, hs_idx):
    # Dataset labels are directory class indices. Count without changing data.
    counts = np.zeros(2, dtype=np.int64)
    for _, labels in train_ds:
        vals, nums = np.unique(labels.numpy().astype(int).ravel(), return_counts=True)
        for v, n in zip(vals, nums): counts[v] += n
    total = counts.sum()
    weights = {int(i): float(total / (2 * max(counts[i], 1))) for i in range(2)}
    print(f"Class counts (class_names order): {counts.tolist()} | class weights: {weights}")
    return weights


def run(benchmark):
    print(f"\n=== Training ICCAD-{benchmark} ===")
    train_ds, train_classes = load_split(benchmark, "train", shuffle=True)
    val_ds, val_classes = load_split(benchmark, "validation", shuffle=False)
    if train_classes != val_classes:
        raise ValueError(f"Class folder order differs: train={train_classes}, validation={val_classes}")
    hs_idx = hotspot_index(train_classes)
    model = build_model()
    class_weight = calculate_class_weights(train_ds, hs_idx) if USE_CLASS_WEIGHTS else None
    # Keep the epoch count fixed: validation is reserved for temperature fitting
    # and threshold selection, so it must not choose the training epoch.
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                     class_weight=class_weight, verbose=1)
    model_path = OUTPUT_DIR / f"model_benchmark_{benchmark}.keras"
    model.save(model_path)
    metadata = {"benchmark": benchmark, "class_names": train_classes,
                "hotspot_class_index": hs_idx, "image_size": list(IMG_SIZE),
                "output_is_logit_for_class_index_1": True,
                "epochs_completed": len(hist.history.get("loss", []))}
    (OUTPUT_DIR / f"model_benchmark_{benchmark}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {model_path}; validation accuracy shown during training only.")

if __name__ == "__main__":
    for b in BENCHMARKS: run(b)
    print("Training complete. No test images were loaded in this stage.")
