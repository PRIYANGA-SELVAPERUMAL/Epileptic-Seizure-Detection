# 🧠 Epileptic Seizure Detection using Hybrid CNN–BiLSTM Model

An end-to-end deep learning project that leverages a hybrid CNN–BiLSTM architecture to predict epileptic seizures using EEG signals. The system integrates robust signal processing, model training, and a web application for real-time detection — all built from scratch.

---

## 📌 Table of Contents

* [About the Project](#about-the-project)
* [Dataset](#dataset)
* [Problem Statement](#problem-statement)
* [Methodology](#methodology)
* [Model Architecture](#model-architecture)
* [Website Overview](#website-overview)
* [Tech Stack](#tech-stack)
* [How to Run the Project](#how-to-run-the-project)
* [Results](#results)
* [Future Work](#future-work)

---

## 📖 About the Project

Epileptic seizures are bursts of abnormal electrical brain activity. Manual diagnosis through EEG interpretation is both time-intensive and expertise-driven. This project automates seizure detection by using a hybrid deep learning model (CNN + BiLSTM) trained on EEG data and provides a simple web interface for real-time predictions.

---

## 📊 Dataset

We used the Epileptic Seizure Recognition Dataset from UCI Machine Learning Repository:

* 11,500 samples
* Each sample has 178 EEG signal points
* 5 Classes originally, remapped to binary:

  * 1 → Seizure
  * 2–5 → Non-seizure

🔗 [Dataset Link](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)

---

## ❓ Problem Statement

Design a hybrid deep learning model that accurately detects epileptic seizures from EEG data and deploy it via a clean web interface to assist patients and healthcare professionals.

---

## 🔍 Methodology

1. Data Cleaning & Preprocessing:

   * Normalization of EEG signals
   * Label consolidation (binary: seizure vs non-seizure)
   * Train-test split

2. Model Architecture:

   * Use of hybrid CNN–BiLSTM model
   * CNN for feature extraction
   * BiLSTM for sequential dependency modeling
   * Output via softmax activation

3. Web Deployment:

   * Model saved as .h5
   * Flask backend for prediction
   * Frontend for uploading and classifying signals

---

## 🧠 Model Architecture: CNN–BiLSTM

* 1D Convolutional Layers: Extract spatial features from EEG sequences
* MaxPooling Layer: Reduce dimensionality
* BiLSTM Layer: Capture temporal dependencies and bidirectional flow
* Fully Connected Layer: Dense interpretation
* Output Layer: Softmax for binary classification

✅ Final Accuracy: 98.61%
📉 Validation Loss: Stable convergence after tuning

---

## 🌐 Website Overview

The project includes a web-based front end built using HTML/CSS and Flask. Users can:

* Upload EEG signal data (CSV format)
* View real-time seizure prediction (Seizure / Non-Seizure)
* Understand classification probability



![image](https://github.com/user-attachments/assets/357514d6-0c3d-4da1-a686-4126433a6529)
![image](https://github.com/user-attachments/assets/1ed47b7f-0570-452c-a46a-6c723a500cfe)


---

## 🛠️ Tech Stack

| Layer         | Tools/Frameworks          |
| ------------- | ------------------------- |
| Front-End     | HTML, CSS, JavaScript     |
| Back-End      | Python, Flask             |
| Deep Learning | TensorFlow/Keras          |
| Data Handling | Pandas, NumPy             |
| Visualization | Matplotlib, Seaborn       |
| Deployment    | Localhost                 |

---

## ⚙️ How to Run the Project

1. Clone the repo:

   ```
   git clone https://github.com/PRIYANGA-SELVAPERUMAL/epileptic-seizure-detection.git
   cd epileptic-seizure-detection
   ```

2. Create and activate virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install requirements:

   ```
   pip install -r requirements.txt
   ```

4. Run Flask server:

   ```
   python app.py
   ```

5. Open browser:

   ```
   http://127.0.0.1:5000/
   ```

---

## ✅ Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 98.61% |
| Precision | 0.98  |
| Recall    | 0.98 |
| F1-Score  | 0.98 |

Confusion matrix, Classification report, and learning curves are available in the repo under the results section.

## 🔭 Future Enhancements

* Integrate real-time EEG device stream
* Deploy to cloud (Render/AWS)
* Improve generalization with transfer learning
* Add mobile app version for accessibility


