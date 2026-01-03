# EE4065 – HW6  
## Handwritten Digit Recognition from Digital Images (Embedded-Oriented)

This repository contains the complete implementation of **Homework 6** for the EE4065 – Embedded Digital Image Processing course.

The homework is based on **Section 13.7 – Application: Handwritten Digit Recognition from Digital Images**, where convolutional neural networks are trained on the MNIST dataset and prepared for **embedded deployment** using **TensorFlow Lite**.

---

## 1. Objective of the Homework

The main objectives of this homework are:

- To implement handwritten digit recognition using CNN-based models  
- To train and evaluate different network architectures on the MNIST dataset  
- To convert trained models into **TensorFlow Lite (.tflite)** format  
- To export models as **C arrays (.cc)** suitable for microcontroller inference  
- To compare models in terms of **accuracy, parameter count, and memory footprint**  

This workflow directly follows the pipeline described in **Listings 13.6 and 13.7** of the course textbook.

---

## 2. Dataset and Preprocessing

- **Dataset:** MNIST handwritten digit dataset  
- **Image size:** Resized to **32 × 32 × 3**  
- **Normalization:** Pixel values scaled to `[0, 1]`  
- **Labels:** One-hot encoded (10 classes)

The preprocessing steps are implemented in `common.py`, ensuring consistency across all models.

---

## 3. Implemented Models

Three different CNN-based approaches were implemented and evaluated:

### 3.1 SmallCNN (Custom CNN)
A lightweight convolutional neural network designed specifically for embedded systems.

- Few convolutional layers
- Low parameter count
- High accuracy on MNIST
- Best memory–performance trade-off

---

### 3.2 MobileNetV2 (Frozen)
- Pretrained MobileNetV2 used as a **fixed feature extractor**
- All convolutional layers frozen
- Only classification layers trained

This approach mimics the use of pretrained networks without fine-tuning, as a baseline comparison.

---

### 3.3 MobileNetV2 (Fine-tuned)
- MobileNetV2 with selected layers unfrozen
- Fine-tuning improves accuracy significantly
- Higher computational and memory cost compared to SmallCNN

---

## 4. Training and Evaluation

All models were trained using the same training–validation split of the MNIST dataset.  
Test accuracy was measured on the official MNIST test set.

After training, each model was:
1. Saved in **Keras (.keras)** format  
2. Converted to **TensorFlow Lite (.tflite)**  
3. Exported as a **C array (.cc)** for embedded deployment  

---

## 5. Results and Comparison

### 5.1 Quantitative Results

| Model                    | Test Accuracy | Parameters (M) | TFLite Size (MB) | Accuracy / MB |
|--------------------------|---------------|----------------|------------------|---------------|
| MobileNetV2 (Frozen)     | 0.6095        | 2.271          | 2.403            | 0.25          |
| MobileNetV2 (Fine-tuned) | 0.9716        | 2.271          | 2.403            | 0.40          |
| SmallCNN                 | 0.9929        | 0.316          | 0.310            | 3.20          |

The metrics were automatically generated using `make_metrics.py` and saved under the `results/` directory.

---

## 6. Discussion and Interpretation

- The **SmallCNN** model achieves the **highest accuracy** while using **significantly fewer parameters**.
- Although **MobileNetV2 (Fine-tuned)** reaches high accuracy, its memory footprint is large for embedded systems.
- **MobileNetV2 (Frozen)** performs poorly due to limited adaptation to the MNIST dataset.

From an **embedded systems perspective**, SmallCNN clearly provides the best balance between:
- Accuracy  
- Model size  
- Memory efficiency  

---

## 7. Embedded Deployment Readiness

Following the methodology in **Listing 13.7**, all trained models were:

- Converted to TensorFlow Lite using default optimizations  
- Exported as C source files (`.cc`)  

These outputs are directly compatible with embedded inference frameworks such as:
- STM32CubeIDE  
- TensorFlow Lite for Microcontrollers  

Actual deployment on STM hardware is not performed in this homework, but the generated outputs are **fully deployment-ready**.

## 8. Repository Structure

```text

HW6/
├── code/ # Training, conversion, and evaluation scripts
├── models/ # Keras, TFLite, and C array models
├── results/ # CSV result files
└── .gitignore

```


---

## 9. Conclusion

This homework successfully implements the complete handwritten digit recognition pipeline described in Section 13.7 of the course material.  
Through comparative evaluation, it is shown that **task-specific lightweight CNNs** outperform large pretrained models in embedded scenarios.

---

## 10. Homework Authors

- **Semih Yıldız** – 150721029  
- **Rüzgar Batı Okay** – 150722048  

---

## 11. Notes

This repository is prepared solely for academic purposes as part of EE4065 coursework.


















## 8. Repository Structure

