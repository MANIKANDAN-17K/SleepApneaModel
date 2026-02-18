import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


# PREPROCESSING FUNCTIONS


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    Bandpass filter to keep ECG relevant frequencies
    - lowcut  : 0.5 Hz  → removes baseline wander
    - highcut : 40.0 Hz → removes high frequency noise
    - order   : filter order
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def notch_filter(signal, fs, freq=50.0, quality=30.0):
    """
    Notch filter to remove powerline interference
    - freq    : 50 Hz (use 60 Hz if in USA)
    - quality : Q factor
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, quality)
    filtered = filtfilt(b, a, signal)
    return filtered


def moving_average_smooth(signal, window_size=5):
    """
    Moving average smoothing
    - window_size : number of samples to average
    """
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, kernel, mode='same')
    return smoothed


def preprocess_ecg(signal, fs):
    """
    Full preprocessing pipeline:
    1. Bandpass filter  → remove baseline wander + high freq noise
    2. Notch filter     → remove powerline interference (50Hz)
    3. Smoothing        → moving average for final cleanup

    Args:
        signal : raw ECG signal (numpy array)
        fs     : sampling frequency (Hz)

    Returns:
        signal_clean : preprocessed ECG signal (numpy array)
    """
    # Step 1 — Bandpass Filter
    signal_bp = bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0)

    # Step 2 — Notch Filter
    signal_notch = notch_filter(signal_bp, fs, freq=50.0)

    # Step 3 — Moving Average Smoothing
    signal_clean = moving_average_smooth(signal_notch, window_size=5)

    return signal_clean



# QUICK TEST (only runs when file is executed directly)


if __name__ == "__main__":
    import wfdb
    import os
    import matplotlib.pyplot as plt

    DATA_PATH = "/home/manikandan/Projects/apnea-ecg-database-1.0.0/"

    # Load test record
    record_path = os.path.join(DATA_PATH, "a01")
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]
    fs = record.fs

    # Preprocess
    signal_clean = preprocess_ecg(signal, fs)

    print(f"Original signal  — min: {signal.min():.4f}, max: {signal.max():.4f}")
    print(f"Processed signal — min: {signal_clean.min():.4f}, max: {signal_clean.max():.4f}")
    print("Preprocessing test passed!")

    # Plot
    time = np.arange(3000) / fs
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    axes[0].plot(time, signal[:3000], color='gray', linewidth=0.8)
    axes[0].set_title("Raw ECG Signal")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True)

    signal_bp = bandpass_filter(signal, fs)
    axes[1].plot(time, signal_bp[:3000], color='blue', linewidth=0.8)
    axes[1].set_title("After Bandpass Filter (0.5 - 40 Hz)")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].grid(True)

    signal_notch = notch_filter(signal_bp, fs)
    axes[2].plot(time, signal_notch[:3000], color='green', linewidth=0.8)
    axes[2].set_title("After Notch Filter (50 Hz removed)")
    axes[2].set_ylabel("Amplitude (mV)")
    axes[2].grid(True)

    axes[3].plot(time, signal_clean[:3000], color='red', linewidth=0.8)
    axes[3].set_title("After Moving Average Smoothing (Final Clean Signal)")
    axes[3].set_ylabel("Amplitude (mV)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].grid(True)

    plt.suptitle("ECG Preprocessing Pipeline — a01", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("ecg_preprocessing.png", dpi=150)
    plt.show()
    print("Plot saved as ecg_preprocessing.png")