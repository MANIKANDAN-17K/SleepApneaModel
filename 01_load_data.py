import wfdb 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
 
# Dataset Path  
DATA_PATH = "/home/manikandan/Projects/apnea-ecg-database-1.0.0/" 
 
# Record Groups  
class_a = [f"a{i:02d}" for i in range(1, 21)]   # a01 - a20 
class_b = [f"b{i:02d}" for i in range(1, 6)]    # b01 - b05 
class_c = [f"c{i:02d}" for i in range(1, 11)]   # c01 - c10 
test_set = [f"x{i:02d}" for i in range(1, 36)]  # x01 - x35 
 
learning_set = class_a + class_b + class_c 
 
print(f"Apnea Records      : {len(class_a)}") 
print(f"Borderline Records : {len(class_b)}") 
print(f"Control Records    : {len(class_c)}") 
print(f"Test Records       : {len(test_set)}") 
print(f"Total Records      : {len(learning_set) + len(test_set)}") 
 
# Load Single Record  
def load_ecg(record_name): 
    record_path = os.path.join(DATA_PATH, record_name) 
    record = wfdb.rdrecord(record_path) 
    signal = record.p_signal[:, 0]  # ECG channel 
    fs = record.fs                   # sampling frequency 
    return signal, fs 
 
# Load Apnea Annotations  
def load_annotations(record_name): 
    record_path = os.path.join(DATA_PATH, record_name) 
    ann = wfdb.rdann(record_path, 'apn') 
    labels = ann.symbol   # 'A' = Apnea, 'N' = Normal 
    samples = ann.sample  # sample index of each annotation 
    return labels, samples 
 
# Test Loading One Record  
record_name = "a01" 
signal, fs = load_ecg(record_name) 
labels, samples = load_annotations(record_name) 
 
print(f"\n===== Record Information: {record_name} =====") 
print(f"ECG Signal Shape : {signal.shape}") 
print(f"Sampling Rate    : {fs} Hz") 
print(f"Recording Length : {len(signal)/fs/3600:.2f} hours") 
print(f"Annotations      : {len(labels)}") 
print(f"Apnea Minutes    : {labels.count('A')}") 
print(f"Normal Minutes   : {labels.count('N')}") 
 
# Quick Plot  
plt.figure(figsize=(15, 4)) 
time = np.arange(len(signal)) / fs 
plt.plot(time[:3000], signal[:3000], color='blue', linewidth=0.8) 
plt.title(f"ECG Signal - {record_name} (First 30 seconds)") 
plt.xlabel("Time (seconds)") 
plt.ylabel("Amplitude (mV)") 
plt.grid(True) 
plt.tight_layout() 
plt.show()
