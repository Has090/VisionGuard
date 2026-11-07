import tensorflow as tf
from tensorflow.keras import layers, models

def build_baseline_cnn(input_shape=(224, 224, 3)):
    model = models.Sequential([
        # 1. Rescaling layer: maps [0, 255] byte pixels to [0.0, 1.0] floats
        layers.Rescaling(1.0 / 255.0, input_shape=input_shape, name="rescaling"),

        # Block 1: Low-level edge and texture extraction
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling2D((2, 2), name="pool1"),

        # Block 2: Intermediate structural and localized contour features
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D((2, 2), name="pool2"),

        # Block 3: Complex compositional and texture anomaly patterns
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.MaxPooling2D((2, 2), name="pool3"),

        # Block 4: High-level semantic abstractions
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv4"),
        layers.BatchNormalization(name="bn4"),
        layers.MaxPooling2D((2, 2), name="pool4"),

        # Classification Head: Global pooling minimizes parameters vs standard Flatten
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dropout(0.4, name="dropout"),
        layers.Dense(64, activation="relu", name="dense_features"),
        layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="VisionGuard_Baseline_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc")
        ]
    )
    return model

if __name__ == "__main__":
    model = build_baseline_cnn()
    model.summary()

    # Formatted Markdown summary table for report
    print("\n" + "="*80)
    print("BASELINE CNN LAYER PARAMETER & JUSTIFICATION TABLE")
    print("="*80)
    print("| Layer Name | Type | Output Shape | Parameters | Functional Justification |")
    print("|---|---|---|---|---|")
    
    justifications = {
        "rescaling": "Normalizes pixel dynamic range from [0, 255] to [0.0, 1.0] for gradient stability.",
        "conv1": "Extracts 32 low-level spatial kernels targeting edges, fine lines, and color gradients.",
        "bn1": "Standardizes layer activations, smoothing the optimization landscape.",
        "pool1": "Downsamples spatial grid by 2x, providing translation invariance.",
        "conv2": "Extracts 64 mid-level shape, curve, and brush-stroke texture filters.",
        "bn2": "Mitigates internal covariate shift across training batches.",
        "pool2": "Reduces dimension from 112x112 to 56x56.",
        "conv3": "Extracts 128 higher-level structural feature maps.",
        "bn3": "Maintains zero-mean unit-variance activations to avoid vanishing gradients.",
        "pool3": "Reduces dimension from 56x56 to 28x28.",
        "conv4": "Learns high-level synthetic generative artifact patterns.",
        "bn4": "Stabilizes activation outputs before dimensional collapse.",
        "pool4": "Reduces dimension from 28x28 to 14x14.",
        "gap": "Aggregates spatial dimensions to a 1D vector (128 units), eliminating parameter explosion.",
        "dropout": "Randomly zeros 40% of neuron activations during training to reduce co-adaptation.",
        "dense_features": "Learns non-linear combinations of pooled spatial features.",
        "output_sigmoid": "Maps logits to a single continuous probability score [0.0, 1.0]."
    }

    for lyr in model.layers:
        p_count = lyr.count_params()
        shape = str(lyr.output_shape)
        just = justifications.get(lyr.name, "Feature transformation.")
        print(f"| `{lyr.name}` | `{lyr.__class__.__name__}` | `{shape}` | {p_count:,} | {just} |")
    print("="*80)