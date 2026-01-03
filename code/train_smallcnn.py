import tensorflow as tf
from common import load_and_prepare_mnist, IMG_SHAPE, NUM_CLASSES

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=IMG_SHAPE),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

def main():
    x_train, y_train, x_test, y_test = load_and_prepare_mnist()

    model = build_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    cb = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=15,
        batch_size=128,
        callbacks=cb,
        verbose=1
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    model.save("../models/smallcnn_mnist.keras")
    print("Saved: ../models/smallcnn_mnist.keras")

if __name__ == "__main__":
    main()
