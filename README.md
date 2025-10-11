# Morph Image Detection — Version 3

This repository presents a deep learning-based system for **morph image detection**, designed to identify whether a facial image is **real** or **synthetically morphed**.
The project leverages the **YOLOv8 classification framework**, fine-tuned through iterative training on custom datasets, with a focus on **temporal coherence, generalization, and real-world applicability**.

A **Streamlit-based graphical interface** is included for seamless offline testing and visualization.
This version (v3) represents a stabilized and optimized checkpoint in the project’s development cycle.

---

## 1. Overview

Morphing attacks pose a serious threat to face-based identity systems such as passport verification, eKYC, and border security.
These attacks are created by blending multiple facial images into one synthetic image that visually resembles both subjects, allowing impersonation during biometric verification.

The goal of this project is to develop a robust and practical morph detection model that can:

* Accurately detect morphing artifacts across multiple datasets.
* Operate efficiently on both CPU and GPU environments.
* Provide a user-friendly interface for testing and demonstration.

The model combines the strengths of **YOLOv8’s feature extraction** with **custom hard-mining cycles**, enabling improved discrimination between authentic and morphed faces.

---

## 2. Key Features

* **YOLOv8 Classification Backbone:** Utilizes pretrained weights for transfer learning, enhancing feature sensitivity to subtle morph artifacts.
* **Offline Inference via Streamlit GUI:** Allows local execution without internet access or cloud dependencies.
* **GPU Acceleration:** Automatically detects CUDA for faster performance.
* **Uncertainty Classification:** Samples below a confidence threshold (default: 0.80) are flagged as "uncertain."
* **Iterative Hard Mining:** Misclassified and ambiguous samples from previous rounds are reintroduced for fine-tuning.
* **Auto-Saving Results:** Detection results are stored as timestamped CSVs for easy analysis and record-keeping.

---

## 3. Project Structure

```
morph_image_detection_github/
    ├── morph_gui_v3.py  #Simple GUI created using streamlit to run and test the model
    |--.gitignore
    |---yolo_cv_runs/production/v3
        |--v3
           |--morphdet_v3.pt    #YOLOv8 classifier model (Size:9.18 MB)
```

---

## 4. Training Methodology

The training pipeline was designed to improve **generalization and robustness** against overfitting to specific datasets or identities.

### 4.1 Base Model

The **YOLOv8 classification model** was initialized with pretrained ImageNet weights and fine-tuned for binary classification:

* **Class 0:** Real face images
* **Class 1:** Morphed face images

The classifier’s architecture retained the convolutional backbone while replacing the output layer with a two-class softmax head.

### 4.2 Data Preprocessing

Images were standardized to **224×224 resolution** and normalized between 0–1.
Strong **data augmentations** were applied to prevent overfitting and simulate real-world variability:

* Color jittering
* Gaussian blur and compression noise
* Random rotations and flips
* Brightness and contrast adjustment

### 4.3 Training Stages

The training was conducted in **five active learning rounds**, each refining the model using hard-mined samples:

1. **Initial Training:** On a balanced dataset of real and morph samples.
2. **Hard Mining Loop:** Misclassified and uncertain samples from evaluation rounds were collected and retrained.
3. **Fine-Tuning:** YOLOv8 was fine-tuned with a smaller learning rate (1e−5) using the universal mixed dataset.
4. **Evaluation:** External test sets were used to assess performance.
5. **Final Model Export:** The best checkpoint (`best.pt`) was saved as `morphdet_v3.pt`.

Training was executed using **PyTorch on CUDA**, with batch sizes adjusted for GPU memory efficiency.

---

## 5. Datasets Used

A combination of **internal** and **external** datasets was used to ensure diversity and generalization.

### 5.1 Internal Datasets

* **CNN Dataset:** Custom dataset created for controlled morph and real samples.
* **External Small Dataset:** A secondary dataset used for balancing morph and real samples.
* **Hardmine Dataset:** Contains misclassified or ambiguous samples from previous rounds, crucial for iterative improvement.

### 5.2 External Benchmark Dataset

* **LFW (Labeled Faces in the Wild):**
  Used as an external validation set containing unconstrained real-world facial images. Synthetic morphs were generated from LFW pairs using face blending algorithms.

### 5.3 Final Dataset Composition

| Category  | Source Datasets            | Total Images                  |
| --------- | -------------------------- | ----------------------------- |
| Real      | CNN, LFW, External Small   | ~3000                         |
| Morph     | Synthetic + Hardmine + CNN | ~5000                         |
| Uncertain | -                          | 0 (excluded in final version) |

The combined dataset of approximately **8000 images** was balanced and stored under
`universal_dataset_pool/` for training and evaluation.

---

## 6. Model Evaluation

After each training phase, the model was evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**
* **Classification Report**

Hard-mined samples were extracted where the model’s confidence was below 0.80.
This continuous feedback loop significantly improved the overall model reliability in later rounds.

Final evaluation results for v3 demonstrated consistent improvements in morph detection rates, particularly in previously uncertain cases.

---

## 7. Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/SibiAathithyas/morph-image-detection.git
   cd morph-image-detection
   ```

2. **Set Up the Environment**

   ```bash
   conda create -n morphdetect_gpu python=3.9 -y
   conda activate morphdetect_gpu
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit GUI**

   ```bash
   streamlit run morph_gui_v3.py
   ```

---

## 8. Usage

Once the interface launches:

1. Upload one or multiple `.jpg` / `.png` images.
2. The model performs inference and displays predictions for each image.
3. Each output includes:

   * Image preview
   * Predicted label (`Real`, `Morph`, or `Uncertain`)
   * Confidence score
4. Results are automatically saved in a timestamped CSV under:

   ```
   yolo_cv_runs/test_eval/gui_cumulative_<timestamp>/
   ```

---

## 9. Technical Summary

| Parameter                     | Description                      |
| ----------------------------- | -------------------------------- |
| **Model Type**                | YOLOv8 Classification            |
| **Classes**                   | 2 (Real, Morph)                  |
| **Input Size**                | 224 × 224                        |
| **Framework**                 | PyTorch / Ultralytics YOLO       |
| **Optimizer**                 | Adam                             |
| **Learning Rate**             | 1e−4 → 1e−5 (during fine-tuning) |
| **Batch Size**                | 16                               |
| **Loss Function**             | Cross-Entropy                    |
| **Threshold for Uncertainty** | 0.80                             |
| **GPU Support**               | Enabled (CUDA)                   |

---

## 10. Dependencies

All required libraries are listed in the `requirements.txt` file.
Major dependencies include:

* ultralytics
* torch
* streamlit
* pandas
* Pillow
* numpy

---

## 11. Contributors

**Lead Developer:** Sibi Aathithyaa
**Project Title:** Morph Image Detection using YOLOv8 and Streamlit GUI
**Collaborators:** App development and deployment team

---

## 12. Future Work

* Development of an Android/iOS mobile application for real-time morph detection.
* Integration with government or institutional ID systems for verification.
* Implementation of adaptive thresholding for uncertainty handling.
* Exploring transformer-based or hybrid CNN architectures for further performance gains.

---

## 13. License and Usage

This repository is intended for **academic and educational use**.
The model, data, and code may be reused or extended for non-commercial research,
provided with proper permission and attribution is maintained to the original authors.
