# DLGenAI NPPE 1 – Code

**Name:** Rajeev Patel  
**Roll Number:** 23f3002112

---

## Complete Implementation Code

The following is the full code used in the Kaggle notebook `23f3002112-notebook-26T1`.

---

### Cell 1 – Imports

```python
import os, random, gc
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
```

---

### Cell 2 – Configuration

```python
class CFG:
    SEED = 42
    IMG_SIZE = 224          # ResNet standard input size
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3
    NUM_WORKERS = 4
    TRAIN_SIZE = 25000      # stratified subset to keep runtime manageable

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_DIR = "/kaggle/input/competitions/26-t-1-dl-gen-ainppe-1"
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    MODEL_PATH = "/kaggle/working/best_model.pth"
    SUB_PATH   = "/kaggle/working/submission.csv"
```

---

### Cell 3 – Reproducibility Seed

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.SEED)
print("Device:", CFG.DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

### Cell 4 – Load and Prepare Data

```python
train_df = pd.read_csv(CFG.TRAIN_CSV)
test_df  = pd.read_csv(CFG.TEST_CSV)

id_col     = "id"
class_cols = [c for c in train_df.columns if c != id_col]   # 20 class columns

# Convert one-hot labels to integer class index
train_df["label"]      = train_df[class_cols].values.argmax(axis=1)
train_df["image_path"] = train_df["id"].apply(lambda x: os.path.join(CFG.IMAGE_DIR, x))
test_df["image_path"]  = test_df["id"].apply(lambda x: os.path.join(CFG.IMAGE_DIR, x))

print("Train:", train_df.shape)
print("Test :", test_df.shape)
print("Classes:", len(class_cols))
```

---

### Cell 5 – Stratified Train/Validation Split

```python
# Draw a smaller stratified subset to control runtime
train_df_small, _ = train_test_split(
    train_df,
    train_size=CFG.TRAIN_SIZE,
    random_state=CFG.SEED,
    stratify=train_df["label"]
)

# 85% train, 15% validation
train_data, val_data = train_test_split(
    train_df_small,
    test_size=0.15,
    random_state=CFG.SEED,
    stratify=train_df_small["label"]
)

train_data = train_data.reset_index(drop=True)
val_data   = val_data.reset_index(drop=True)

print("Train split:", train_data.shape)
print("Val split:",   val_data.shape)
```

---

### Cell 6 – Image Transforms

```python
# Training transforms: resize + random horizontal flip + normalise
train_tfms = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

# Validation/test transforms: resize + normalise only (no augmentation)
valid_tfms = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
```

---

### Cell 7 – Custom Dataset Class

```python
class XrayDataset(Dataset):
    def __init__(self, df, transform=None, test=False):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.test      = test   # if True, returns image id instead of label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        if self.test:
            return img, row["id"]
        return img, int(row["label"])
```

---

### Cell 8 – DataLoaders

```python
train_ds = XrayDataset(train_data, transform=train_tfms, test=False)
val_ds   = XrayDataset(val_data,   transform=valid_tfms, test=False)
test_ds  = XrayDataset(test_df,    transform=valid_tfms, test=True)

train_loader = DataLoader(
    train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
    num_workers=CFG.NUM_WORKERS, pin_memory=True
)

val_loader = DataLoader(
    val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
    num_workers=CFG.NUM_WORKERS, pin_memory=True
)

test_loader = DataLoader(
    test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
    num_workers=CFG.NUM_WORKERS, pin_memory=True
)

print(len(train_loader), len(val_loader), len(test_loader))
```

---

### Cell 9 – Class Weights for Imbalanced Data

```python
# Compute inverse-frequency weights so rare classes get higher loss weight
counts  = train_data["label"].value_counts().sort_index().values
weights = 1.0 / counts
weights = weights / weights.sum() * len(weights)   # normalise to mean = 1
weights = torch.tensor(weights, dtype=torch.float32)

if CFG.DEVICE == "cuda":
    weights = weights.to(CFG.DEVICE)

print("Class weights ready")
```

---

### Cell 10 – Model Definition

```python
def build_model(num_classes):
    # Load ResNet-18 with ImageNet pre-trained weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace the final fully-connected layer for our 20-class task
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model     = build_model(len(class_cols)).to(CFG.DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)          # weighted loss
optimizer = optim.AdamW(model.parameters(), lr=CFG.LR)
scaler    = torch.cuda.amp.GradScaler(enabled=(CFG.DEVICE == "cuda"))  # AMP
```

---

### Cell 11 – Competition Metric

```python
def comp_score(y_true, y_pred, num_classes):
    """
    Custom competition metric:
      per-class score = (TP - FP - 5*FN) / N
      final score     = mean over all classes
    Heavily penalises false negatives (weight = 5).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = []

    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)

        tp = np.sum(true_c & pred_c)
        fp = np.sum(~true_c & pred_c)
        fn = np.sum(true_c & ~pred_c)
        n  = np.sum(true_c)

        score_c = 0 if n == 0 else (tp - fp - 5 * fn) / n
        scores.append(score_c)

    return float(np.mean(scores))
```

---

### Cell 12 – Training Loop

```python
best_score = -9999

for epoch in range(CFG.EPOCHS):
    # ---- Training phase ----
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{CFG.EPOCHS}")
    for images, labels in pbar:
        images = images.to(CFG.DEVICE, non_blocking=True)
        labels = labels.to(CFG.DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Mixed-precision forward pass
        with torch.cuda.amp.autocast(enabled=(CFG.DEVICE == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # ---- Validation phase ----
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{CFG.EPOCHS}"):
            images = images.to(CFG.DEVICE, non_blocking=True)
            labels = labels.to(CFG.DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(CFG.DEVICE == "cuda")):
                outputs = model(images)

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    score    = comp_score(y_true, y_pred, len(class_cols))
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}  val_score={score:.5f}")

    # Save the best model checkpoint
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), CFG.MODEL_PATH)
        print("Best model saved")

    # Free GPU memory between epochs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Best validation score:", best_score)
```

---

### Cell 13 – Test Inference

```python
# Load the best checkpoint for inference
model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location=CFG.DEVICE))
model.eval()

pred_ids    = []
pred_labels = []

with torch.no_grad():
    for images, ids in tqdm(test_loader, desc="Predict Test"):
        images = images.to(CFG.DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(CFG.DEVICE == "cuda")):
            outputs = model(images)

        preds = outputs.argmax(dim=1).cpu().numpy()

        pred_ids.extend(ids)
        pred_labels.extend(preds)
```

---

### Cell 14 – Build and Save Submission

```python
# Build one-hot submission from predicted class indices
submission     = pd.DataFrame(0, index=np.arange(len(pred_ids)), columns=["id"] + class_cols)
submission["id"] = pred_ids

for i, p in enumerate(pred_labels):
    submission.loc[i, class_cols[p]] = 1   # set predicted class column to 1

# Sanity check: exactly one 1 per row
assert (submission[class_cols].sum(axis=1) == 1).all()

submission.to_csv(CFG.SUB_PATH, index=False)
print("Saved:", CFG.SUB_PATH)
submission.head()
```

