import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

from dataset import load_and_split
from model import PatchTST, PatchTSTConfig

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

EPOCHS      = 50
LR          = 1e-4
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH   = "best_model.pth"

print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────

train_loader, val_loader, test_loader, class_weights = load_and_split(
    x_path="X_segments.npy",
    y_path="y_labels.npy"
)

# ──────────────────────────────────────────────────────────
# MODEL, LOSS, OPTIMIZER
# ──────────────────────────────────────────────────────────

cfg   = PatchTSTConfig()
model = PatchTST(cfg).to(DEVICE)

# Weighted loss for class imbalance
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE)
)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ──────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ──────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds  = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


# ──────────────────────────────────────────────────────────
# VALIDATE ONE EPOCH
# ──────────────────────────────────────────────────────────

def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


# ──────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────

print("=" * 60)
print("  Training PatchTST Model")
print("=" * 60)

history = {
    "train_loss": [], "val_loss": [],
    "train_acc" : [], "val_acc" : [],
    "train_f1"  : [], "val_f1"  : []
}

best_val_f1  = 0.0
best_epoch   = 0

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
    val_loss,   val_acc,   val_f1   = val_epoch(model,   val_loader,   criterion)

    scheduler.step()

    # Save history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        torch.save(model.state_dict(), SAVE_PATH)
        saved = "✅ Saved"
    else:
        saved = ""

    print(f"Epoch [{epoch:02d}/{EPOCHS}] "
          f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} "
          f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} "
          f"| LR: {scheduler.get_last_lr()[0]:.6f} {saved}")

print(f"\n✅ Best model saved at epoch {best_epoch} with Val F1: {best_val_f1:.4f}")

# ──────────────────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
axes[0].plot(history["val_loss"],   label="Val Loss",   color="red")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy
axes[1].plot(history["train_acc"], label="Train Acc", color="blue")
axes[1].plot(history["val_acc"],   label="Val Acc",   color="red")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

# F1 Score
axes[2].plot(history["train_f1"], label="Train F1", color="blue")
axes[2].plot(history["val_f1"],   label="Val F1",   color="red")
axes[2].set_title("F1 Score (Macro)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("F1 Score")
axes[2].legend()
axes[2].grid(True)

plt.suptitle("PatchTST Training History", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()
print("✅ Plot saved as training_history.png")

# ──────────────────────────────────────────────────────────
# FINAL EVALUATION ON TEST SET
# ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Final Evaluation on Test Set")
print("=" * 60)

# Load best model
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        outputs = model(X_batch)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print(classification_report(
    all_labels, all_preds,
    target_names=["Normal", "Apnea"]
))

# Save predictions
np.save("test_preds.npy",  np.array(all_preds))
np.save("test_labels.npy", np.array(all_labels))
print("✅ Saved → test_preds.npy")
print("✅ Saved → test_labels.npy")