import keras_tuner as kt
import tensorflow as tf

def build_toy_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=hp.Int("units", min_value=32, max_value=128, step=32),
            activation="relu"
        ),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    
    hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    # Load built-in Fashion-MNIST dataset
    (x_tr, y_tr), (x_va, y_va) = tf.keras.datasets.fashion_mnist.load_data()
    # Normalize pixel values and use subsets for fast demonstration
    x_tr, x_va = x_tr[:3000] / 255.0, x_va[:1000] / 255.0
    y_tr, y_va = y_tr[:3000], y_va[:1000]

    tuner = kt.Hyperband(
        build_toy_model,
        objective="val_accuracy",
        max_epochs=4,
        directory="reports/tuner_toy",
        project_name="toy_fashion_tuning",
        overwrite=True
    )

    print("\n")
    print("STARTING TOY LAB: KERAS TUNER ON FASHION-MNIST")
    print("\n")
    tuner.search(x_tr, y_tr, validation_data=(x_va, y_va), epochs=4)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n")
    print("KERAS TUNER TOY LAB RESULTS (Fashion-MNIST)")
    print("\n")
    print(f"Optimal Dense Units  : {best_hp.get('units')}")
    print(f"Optimal Learning Rate: {best_hp.get('learning_rate')}")
    print("\n")