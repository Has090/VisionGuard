import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from task2_1_pipeline import build_tf_dataset, SEED, IMAGE_SIZE
from task2_2_baseline_cnn import build_baseline_cnn

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODELS_DIR = Path("models")
FIG_DIR = Path("figures")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv("reports/splits/train.csv")
val_df = pd.read_csv("reports/splits/val.csv")

train_ds = build_tf_dataset(train_df, is_training=True)
val_ds = build_tf_dataset(val_df, is_training=False)

def build_callbacks(model_name):
    checkpoint_file = MODELS_DIR / f"{model_name}_best.keras"
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_file),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        )
    ]

# 3. Train Baseline CNN
print("\n STAGE 1: TRAINING BASELINE CNN FROM SCRATCH \n")
baseline_cnn = build_baseline_cnn()
hist_baseline = baseline_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=build_callbacks("baseline_cnn"),
    verbose=1
)

# 4. Transfer Learning (MobileNetV2)
print("\n")
print("STAGE 2: TRANSFER LEARNING - MOBILENETV2 (FROZEN BACKBONE)")
print("=\n")
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
backbone.trainable = False  # Freeze ImageNet weights completely

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
# MobileNetV2 requires inputs normalized to [-1, 1]
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = backbone(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="tl_gap")(x)
x = tf.keras.layers.Dropout(0.3, name="tl_dropout")(x)
x = tf.keras.layers.Dense(64, activation="relu", name="tl_dense")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="tl_output")(x)

tl_model = tf.keras.Model(inputs, outputs, name="VisionGuard_MobileNetV2")
tl_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")]
)

hist_tl_stage1 = tl_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=build_callbacks("mobilenetv2_stage1"),
    verbose=1
)

print("\n")
print("STAGE 3: TRANSFER LEARNING - FINE-TUNING (UNFREEZING LAST BLOCK)")
print("\n")
# Unfreeze only the final inverted residual block (last 15 layers)
backbone.trainable = True
for layer in backbone.layers[:-15]:
    layer.trainable = False

num_trainable = sum([1 for l in backbone.layers if l.trainable])
print(f"Total layers in MobileNetV2 backbone : {len(backbone.layers)}")
print(f"Trainable layers unfrozen for Stage 2 : {num_trainable}")

# Recompile with a significantly lower learning rate for fine-tuning
tl_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")]
)

hist_tl_stage2 = tl_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=build_callbacks("mobilenetv2_stage2_champion"),
    verbose=1
)

# 5. Curve Plotting Utility
def plot_training_history(history, title, out_filename):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs = range(1, len(loss) + 1)
    best_ep = int(np.argmin(val_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{title} (Best Epoch: {best_ep})", fontsize=13, fontweight="bold")

    ax1.plot(epochs, loss, label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#d62728", lw=2)
    ax1.axvline(best_ep, color="black", linestyle="--", label=f"Best Epoch ({best_ep})")
    ax1.set_title("Binary Crossentropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, acc, label="Train Accuracy", color="#1f77b4", lw=2)
    ax2.plot(epochs, val_acc, label="Val Accuracy", color="#2ca02c", lw=2)
    ax2.axvline(best_ep, color="black", linestyle="--", label=f"Best Epoch ({best_ep})")
    ax2.set_title("Classification Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / out_filename
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"Saved: {out_path}")

plot_training_history(hist_baseline, "Baseline CNN Training Curves", "task2_baseline_curves.png")
plot_training_history(hist_tl_stage1, "MobileNetV2 (Stage 1: Frozen Backbone)", "task2_mobilenet_stage1_curves.png")
plot_training_history(hist_tl_stage2, "MobileNetV2 (Stage 2: Fine-Tuning)", "task2_mobilenet_stage2_curves.png")