import sys
from pathlib import Path
import tensorflow as tf

def convert(model_path: str, out_path: str):
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    Path(out_path).write_bytes(tflite_model)
    print(f"Saved: {out_path} ({len(tflite_model)} bytes)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_tflite.py <in_model.keras> <out_model.tflite>")
        raise SystemExit(1)

    convert(sys.argv[1], sys.argv[2])
