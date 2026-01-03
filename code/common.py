import tensorflow as tf

IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

def load_and_prepare_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    x_train = tf.repeat(x_train, repeats=3, axis=-1)
    x_test = tf.repeat(x_test, repeats=3, axis=-1)

    x_train = tf.image.resize(x_train, IMG_SHAPE[:2]) / 255.0
    x_test = tf.image.resize(x_test, IMG_SHAPE[:2]) / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test
