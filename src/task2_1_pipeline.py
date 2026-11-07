
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Enforce deterministic behavior across libraries
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

SPLIT_DIR = Path("reports/splits")
FIG_DIR = Path("figures")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def generate_splits():
    audit_csv = Path("reports/clean_dataset_audit.csv")
    if not audit_csv.exists():
        raise FileNotFoundError("Missing reports/clean_dataset_audit.csv. Complete Task 1.2 first.")
    
    df = pd.read_csv(audit_csv)
    class_map = {"RealArt": 0, "AiArtData": 1}
    df["label"] = df["class"].map(class_map)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
    )

    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    print(f"Dataset Partitions Created:")
    print(f"  Training Set   : {len(train_df)} samples")
    print(f"  Validation Set : {len(val_df)} samples")
    print(f"  Test Set       : {len(test_df)} samples")
    return train_df, val_df, test_df

augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.1, seed=SEED),
    tf.keras.layers.RandomZoom(0.1, seed=SEED),
    tf.keras.layers.RandomContrast(0.1, seed=SEED),
], name="training_augmentations")

def load_and_preprocess_image(file_path, label):
    raw_bytes = tf.io.read_file(file_path)
    # Decode to 3 RGB channels (handles both RGBA and Grayscale gracefully)
    img = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img.set_shape([IMAGE_SIZE[0],IMAGE_SIZE[1], 3])
    return img, label

def build_tf_dataset(df, is_training=False):
    paths = df["file_path"].values
    labels = df["label"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    
    # Deterministic decoding and resizing
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply spatial & chromatic augmentation only to training partition
    if is_training:
        ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def visualize_augmented_batch(train_ds):
    for images, labels in train_ds.take(1):
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        fig.suptitle("tf.data Pipeline: Sample Augmented Training Batch", fontsize=14, fontweight="bold")
        for idx, ax in enumerate(axes.flatten()):
            img = np.clip(images[idx].numpy() / 255.0, 0.0, 1.0)
            lbl = "AiArtData" if labels[idx].numpy() == 1.0 else "RealArt"
            ax.imshow(img)
            ax.set_title(f"Target: {lbl}", fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        out_path = FIG_DIR / "task2_1_augmented_batch.png"
        plt.savefig(out_path, dpi=250)
        plt.close()
        print(f"Augmented batch figure saved to: {out_path}")

if __name__ == "__main__":
    train_df, val_df, test_df = generate_splits()
    train_ds = build_tf_dataset(train_df, is_training=True)
    visualize_augmented_batch(train_ds)