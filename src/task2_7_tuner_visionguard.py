import keras_tuner as kt
import tensorflow as tf
import pandas as pd
from task2_1_pipeline import build_tf_dataset, IMAGE_SIZE

train_df = pd.read_csv("reports/splits/train.csv")
val_df = pd.read_csv("reports/splits/val.csv")

train_ds = build_tf_dataset(train_df, is_training=True)
val_ds = build_tf_dataset(val_df, is_training=False)

def build_visionguard_tuner(hp):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )

    # 1. Hyperparameter: Number of unfrozen layers (0, 10, or 20)
    unfreeze_count = hp.Choice("unfrozen_layers", values=[0, 10, 20])
    if unfreeze_count == 0:
        backbone.trainable = False
    else:
        backbone.trainable = True
        for layer in backbone.layers[:-unfreeze_count]:
            layer.trainable = False

    # 2. Hyperparameter: Dropout rate (0.2, 0.4, or 0.5)
    dropout_val = hp.Choice("dropout_rate", values=[0.2, 0.4, 0.5])

    # 3. Hyperparameter: Learning rate (1e-3, 1e-4, or 1e-5)
    lr_val = hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5])

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_val)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_val),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    tuner = kt.RandomSearch(
        build_visionguard_tuner,
        objective="val_accuracy",
        max_trials=8,
        directory="reports/tuner_vg",
        project_name="vg_random_search",
        overwrite=True
    )

    print("\n")
    print("STARTING 8 HYPERPARAMETER SEARCH TRIALS FOR VISIONGUARD")
    print("\n")
    tuner.search(train_ds, validation_data=val_ds, epochs=3)

    print("\n")
    print("VISIONGUARD KERAS TUNER - TOP TRIALS SUMMARY")
    print("\n")
    tuner.results_summary(num_trials=8)