🫀 Sleep Apnea Detection Using ECG Signals
📌 Project Overview

This project focuses on detecting Sleep Apnea using overnight ECG recordings.

The system processes long-duration ECG signals, segments them into 1-minute windows, applies signal preprocessing techniques, and trains a deep learning model to classify:

✅ Apnea

✅ Normal breathing

The goal is to build a reliable ECG-based apnea detection pipeline.

🧠 Problem Statement

Sleep apnea is a serious sleep disorder where breathing repeatedly stops and starts.

Manual diagnosis using full polysomnography is:

Expensive

Time-consuming

Requires clinical setup

This project aims to:

Automatically detect apnea using ECG signals only.

📂 Project Structure
SleepApneaModel/
│
├── 01_load_data.py        # Load ECG records and annotations
├── preprocessing.py       # Bandpass filtering and normalization
├── 03_segmentation.py     # Segment ECG into 1-minute windows
├── dataset.py             # PyTorch dataset class
├── model.py               # 1D CNN model definition
├── 06_train.py            # Training pipeline
├── README.md
├── .gitignore

📊 Dataset Information

Sampling rate: 100 Hz

Recording duration: ~8 hours per patient

Labels: Minute-wise annotations

Classes:

Apnea

Normal

Borderline

Control

Each 1-minute segment contains:

100 samples × 60 seconds = 6000 samples

🔬 Preprocessing Pipeline

Load raw ECG signal

Apply Bandpass Filter (0.5 – 40 Hz)

Normalize signal

Segment into 1-minute windows

Assign labels (Apnea / Normal)

🧹 Signal Processing Techniques Used

Bandpass filtering (0.5–40 Hz)

Noise removal

Signal normalization

Segmentation

🤖 Model Architecture

1D Convolutional Neural Network (CNN)

Input: 6000-length ECG segment

Output: Binary classification (Apnea / Normal)

⚙️ Installation

Create virtual environment:

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib scikit-learn wfdb

🚀 How To Run
1️⃣ Load Data
python 01_load_data.py

2️⃣ Segment Data
python 03_segmentation.py

3️⃣ Train Model
python 06_train.py

📈 Evaluation Metrics

Since dataset is imbalanced, we evaluate using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

⚠️ Important Notes

Large .npy files are excluded using .gitignore

Dataset files are not included in this repository

CPU version of PyTorch is used

🧪 Future Improvements

Add LSTM model

Use attention-based models

Perform patient-level cross-validation

Improve class imbalance handling

Deploy as real-time monitoring system

📌 Author

Manikandan K