import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# CONFIGURATION


BATCH_SIZE  = 32
TEST_SIZE   = 0.2
VAL_SIZE    = 0.1
RANDOM_SEED = 42

# DATASET CLASS


class ApneaECGDataset(Dataset):
    """
    PyTorch Dataset for Apnea ECG segments.

    Args:
        segments : numpy array (N, 3000)
        labels   : numpy array (N,)
    """
    def __init__(self, segments, labels):
        # PatchTST expects shape (batch, seq_len, n_vars)
        # We have 1 variable (single ECG channel)
        self.X = torch.tensor(segments, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# LOAD & SPLIT DATA


def load_and_split(x_path="X_segments.npy", y_path="y_labels.npy"):
    """
    Loads saved segments and splits into train/val/test sets.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """

    # Load
    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Loaded X : {X.shape}")
    print(f"Loaded y : {y.shape}")

    # Normalize each window to zero mean, unit variance
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Train / Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train
    )

    print(f"\n── Split Summary ────────────────────────────")
    print(f"Train   : {X_train.shape} | Apnea: {y_train.sum()} | Normal: {(y_train==0).sum()}")
    print(f"Val     : {X_val.shape}   | Apnea: {y_val.sum()}   | Normal: {(y_val==0).sum()}")
    print(f"Test    : {X_test.shape}  | Apnea: {y_test.sum()}  | Normal: {(y_test==0).sum()}")

    # Class weights for imbalance handling
    n_normal = (y_train == 0).sum()
    n_apnea  = (y_train == 1).sum()
    w_normal = 1.0 / n_normal
    w_apnea  = 1.0 / n_apnea
    total    = w_normal + w_apnea
    class_weights = torch.tensor(
        [w_normal / total, w_apnea / total],
        dtype=torch.float32
    )
    print(f"\n── Class Weights ────────────────────────────")
    print(f"Normal weight : {class_weights[0]:.4f}")
    print(f"Apnea  weight : {class_weights[1]:.4f}")

    # Create Datasets
    train_dataset = ApneaECGDataset(X_train, y_train)
    val_dataset   = ApneaECGDataset(X_val,   y_val)
    test_dataset  = ApneaECGDataset(X_test,  y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n── DataLoader Summary ───────────────────────")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_weights


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights = load_and_split()

    # Check one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\n── Sample Batch ─────────────────────────────")
    print(f"X batch shape : {X_batch.shape}  (batch, seq_len, n_vars)")
    print(f"y batch shape : {y_batch.shape}")
    print(f"Class weights : {class_weights}")
    print(f"\nDataset ready for PatchTST!")