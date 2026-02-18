import numpy as np
import wfdb
import os
import sys
sys.path.append("/home/manikandan/Projects/CIP/model/")

from preprocessing import preprocess_ecg


# CONFIGURATION


DATA_PATH = "/home/manikandan/Projects/apnea-ecg-database-1.0.0/"
FS        = 100       # Sampling frequency (Hz)
WIN_SEC   = 30        # Window size in seconds
WIN_SIZE  = FS * WIN_SEC  # 3000 samples per window


# RECORD GROUPS


class_a   = [f"a{i:02d}" for i in range(1, 21)]
class_b   = [f"b{i:02d}" for i in range(1, 6)]
class_c   = [f"c{i:02d}" for i in range(1, 11)]
test_set  = [f"x{i:02d}" for i in range(1, 36)]

learning_set = class_a + class_b + class_c


# LOAD FUNCTIONS


def load_ecg(record_name):
    record_path = os.path.join(DATA_PATH, record_name)
    record      = wfdb.rdrecord(record_path)
    signal      = record.p_signal[:, 0]
    fs          = record.fs
    return signal, fs


def load_annotations(record_name):
    record_path = os.path.join(DATA_PATH, record_name)
    ann         = wfdb.rdann(record_path, 'apn')
    labels      = ann.symbol   # 'A' = Apnea, 'N' = Normal
    samples     = ann.sample   # sample index of each annotation
    return labels, samples


# SEGMENTATION FUNCTION


def segment_record(record_name):
    """
    Segments a single ECG record into 30-second windows
    and assigns label to each window from .apn annotations.

    Each .apn annotation covers 1 minute → 2 windows of 30 sec.
    Both windows inherit the same label from the 1-minute annotation.

    Args:
        record_name : e.g. 'a01', 'b01', 'c01'

    Returns:
        segments : numpy array of shape (N, 3000)
        labels   : numpy array of shape (N,) → 0=Normal, 1=Apnea
    """

    # Load and preprocess signal
    signal, fs  = load_ecg(record_name)
    signal_clean = preprocess_ecg(signal, fs)

    # Load annotations
    ann_labels, ann_samples = load_annotations(record_name)

    segments = []
    labels   = []

    for i, (ann_sample, ann_label) in enumerate(zip(ann_samples, ann_labels)):

        # Each annotation is at the END of 1 minute
        # So the 1-minute window starts 6000 samples before annotation
        minute_start = ann_sample - (FS * 60)
        if minute_start < 0:
            minute_start = 0

        # Split 1 minute into 2 windows of 30 seconds
        for w in range(2):
            win_start = minute_start + (w * WIN_SIZE)
            win_end   = win_start + WIN_SIZE

            # Make sure window is within signal bounds
            if win_end > len(signal_clean):
                break

            window = signal_clean[win_start:win_end]

            # Assign label
            label = 1 if ann_label == 'A' else 0

            segments.append(window)
            labels.append(label)

    segments = np.array(segments)
    labels   = np.array(labels)

    return segments, labels



# PROCESS ALL LEARNING SET RECORDS


def build_dataset(record_list):
    """
    Builds full dataset from a list of records.

    Args:
        record_list : list of record names

    Returns:
        all_segments : numpy array (Total_windows, 3000)
        all_labels   : numpy array (Total_windows,)
        all_records  : list of record names per window
    """
    all_segments = []
    all_labels   = []
    all_records  = []

    for record_name in record_list:
        try:
            segments, labels = segment_record(record_name)
            all_segments.append(segments)
            all_labels.append(labels)
            all_records.extend([record_name] * len(labels))
            print(f"  {record_name} → {len(segments)} windows "
                  f"| Apnea: {labels.sum()} "
                  f"| Normal: {(labels==0).sum()}")
        except Exception as e:
            print(f"  {record_name} → Error: {e}")

    all_segments = np.concatenate(all_segments, axis=0)
    all_labels   = np.concatenate(all_labels,   axis=0)

    return all_segments, all_labels, all_records

# MAIN
if __name__ == "__main__":

    print("=" * 55)
    print("  Building Dataset from Learning Set (35 records)")
    print("=" * 55)

    X, y, records = build_dataset(learning_set)

    print(f"\n── Dataset Summary ──────────────────────────")
    print(f"Total windows    : {len(X)}")
    print(f"Window shape     : {X[0].shape}  (3000 samples = 30 sec)")
    print(f"Apnea windows    : {y.sum()}")
    print(f"Normal windows   : {(y == 0).sum()}")
    print(f"Apnea ratio      : {y.mean() * 100:.1f}%")

    # Save dataset
    np.save("X_segments.npy", X)
    np.save("y_labels.npy",   y)
    print(f"\nSaved → X_segments.npy  shape: {X.shape}")
    print(f" Saved → y_labels.npy    shape: {y.shape}")