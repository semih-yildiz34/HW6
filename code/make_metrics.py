import csv
from pathlib import Path
import tensorflow as tf

MODELS = [
    {
        "name": "mobilenetv2_frozen",
        "keras": "../models/mobilenetv2_mnist.keras",
        "tflite": "../models/mobilenetv2_mnist.tflite",
        "accuracy": 0.6095,
    },
    {
        "name": "mobilenetv2_finetune",
        "keras": "../models/mobilenetv2_finetune_mnist.keras",
        "tflite": "../models/mobilenetv2_finetune_mnist.tflite",
        "accuracy": 0.9716,
    },
    {
        "name": "smallcnn",
        "keras": "../models/smallcnn_mnist.keras",
        "tflite": "../models/smallcnn_mnist.tflite",
        "accuracy": 0.9929,
    },
]

def count_params(model: tf.keras.Model) -> int:
    total = 0
    for w in model.weights:
        total += int(tf.size(w).numpy())
    return total

def bytes_to_kb(b: int) -> float:
    return b / 1024.0

def bytes_to_mb(b: int) -> float:
    return b / (1024.0 * 1024.0)

def main():
    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / "metrics.csv"
    out_pretty = results_dir / "metrics_pretty.csv"

    rows = []
    rows_pretty = []

    for m in MODELS:
        keras_path = Path(m["keras"])
        tflite_path = Path(m["tflite"])

        if not keras_path.exists():
            raise FileNotFoundError(f"Missing keras model: {keras_path}")
        if not tflite_path.exists():
            raise FileNotFoundError(f"Missing tflite model: {tflite_path}")

        model = tf.keras.models.load_model(keras_path)
        params_total = count_params(model)

        tflite_size = tflite_path.stat().st_size

        # Simple "efficiency" score: accuracy per MB (higher is better)
        acc = float(m["accuracy"])
        acc_per_mb = acc / max(bytes_to_mb(tflite_size), 1e-9)

        row = {
            "model": m["name"],
            "test_accuracy": acc,
            "params_total": params_total,
            "tflite_size_bytes": tflite_size,
        }
        rows.append(row)

        row_pretty = {
            "model": m["name"],
            "test_accuracy": f"{acc:.4f}",
            "params_M": f"{params_total/1e6:.3f}",
            "tflite_KB": f"{bytes_to_kb(tflite_size):.1f}",
            "tflite_MB": f"{bytes_to_mb(tflite_size):.3f}",
            "acc_per_MB": f"{acc_per_mb:.2f}",
        }
        rows_pretty.append(row_pretty)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "test_accuracy", "params_total", "tflite_size_bytes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    with out_pretty.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "test_accuracy", "params_M", "tflite_KB", "tflite_MB", "acc_per_MB"]
        )
        writer.writeheader()
        writer.writerows(rows_pretty)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_pretty}")
    for r in rows_pretty:
        print(r)

if __name__ == "__main__":
    main()
