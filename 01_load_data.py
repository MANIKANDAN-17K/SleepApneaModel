import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#Dataset Path 
DATA_PATH = "/home/manikandan/Projects/apnea-ecg-database-1.0.0/"

#  Record Groups 
class_a = [f"a{i:02d}" for i in range(1, 21)]   # a01 - a20
class_b = [f"b{i:02d}" for i in range(1, 6)]    # b01 - b05
class_c = [f"c{i:02d}" for i in range(1, 11)]   # c01 - c10
test_set = [f"x{i:02d}" for i in range(1, 36)]  # x01 - x35

learning_set = class_a + class_b + class_c

print(f"Class A (Apnea)     : {len(class_a)} records")
print(f"Class B (Borderline): {len(class_b)} records")
print(f"Class C (Control)   : {len(class_c)} records")
print(f"Test Set            : {len(test_set)} records")
print(f"Total               : {len(learning_set) + len(test_set)} records")

#  Load Single Record 
def load_ecg(record_name):
    record_path = os.path.join(DATA_PATH, record_name)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]  # ECG channel
    fs = record.fs                   # sampling frequency
    return signal, fs

#  Load Apnea Annotations 
def load_annotations(record_name):
    record_path = os.path.join(DATA_PATH, record_name)
    ann = wfdb.rdann(record_path, 'apn')
    labels = ann.symbol   # 'A' = Apnea, 'N' = Normal
    samples = ann.sample  # sample index of each annotation
    return labels, samples

#  Test Loading One Record 
record_name = "a01"
signal, fs = load_ecg(record_name)
labels, samples = load_annotations(record_name)

print(f"\n── Record: {record_name} ──")
print(f"Signal shape     : {signal.shape}")
print(f"Sampling rate    : {fs} Hz")
print(f"Duration         : {len(signal)/fs/3600:.2f} hours")
print(f"Total annotations: {len(labels)}")
print(f"Apnea minutes    : {labels.count('A')}")
print(f"Normal minutes   : {labels.count('N')}")

#  Quick Plot 
plt.figure(figsize=(15, 4))
time = np.arange(len(signal)) / fs
plt.plot(time[:3000], signal[:3000], color='blue', linewidth=0.8)
plt.title(f"ECG Signal - {record_name} (First 30 seconds)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.tight_layout()
plt.show()