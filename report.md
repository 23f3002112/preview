# 🩺 X-Ray Disease Classification — NPPE 1 Report

<div align="center">

**Rajeev Patel** · Roll No: `23f3002112` · DLGenAI NPPE 1

</div>

---

## 1. Introduction

### 🔍 What is the Problem?

This competition requires **multi-class image classification** of chest X-ray images into one of **20 disease/condition categories**. Each image maps to exactly one class, making it a single-label classification task with one-hot encoded ground-truth labels.

### 💡 Why Does It Matter?

> Automated chest X-ray interpretation can save lives — especially in under-resourced hospitals where specialist radiologists are unavailable. Deep learning models trained on large datasets have matched or exceeded human-level accuracy on several diagnostic tasks, making this a high-stakes, real-world application.

### 📁 The Dataset

| Split | Samples | Classes | Format |
|:------|--------:|--------:|:-------|
| Train | 51,043  | 20      | One-hot CSV + images |
| Test  | 17,015  | —       | CSV + images |

- **Source:** Kaggle competition `26-T-1-DL-GenAI-NPPE-1`
- **Images:** Stored in `images/` directory, referenced by `id` column
- **Labels:** 20 columns (one per class) encoded as one-hot vectors in `train.csv`

### 🎯 Objective

Build a deep learning classifier that **maximises the competition's custom scoring metric**, which is defined as:

$$\text{score}_c = \frac{TP_c - FP_c - 5 \cdot FN_c}{N_c}, \quad \text{Final Score} = \frac{1}{C}\sum_{c=1}^{C} \text{score}_c$$

> ⚠️ The metric penalises **false negatives 5× more** than false positives — strongly favouring high recall over high precision.

---

## 2. Methodology

### 🧠 Model Architecture

**ResNet-18** with ImageNet pre-trained weights was selected as the backbone.

```
Input (224×224×3)
    ↓
ResNet-18 Backbone (frozen feature extractor)
    ↓
Global Average Pooling
    ↓
Linear(512 → 20)   ← replaced final FC layer
    ↓
Softmax → Predicted Class (0–19)
```

**Why ResNet-18?**
- ✅ Lightweight and fast to train on a single T4 GPU within Kaggle's limits
- ✅ Pre-trained ImageNet weights transfer well to medical imagery
- ✅ Strong baseline before scaling to heavier architectures
- ✅ Simple to modify: swap the final FC layer for 20 output classes

---

### ⚙️ Preprocessing Pipeline

```
Raw Image (variable size, RGB)
    │
    ├── Resize → 224×224  (ResNet input standard)
    │
    ├── [Train only] RandomHorizontalFlip(p=0.5)
    │
    ├── ToTensor()  →  [0, 1] float
    │
    └── Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])   ← ImageNet stats
```

> Normalisation with ImageNet statistics is critical when using pre-trained weights — it keeps input activations in the expected distribution the model was trained on.

---

### 🔧 Training Configuration

| Hyperparameter | Value | Reason |
|:---------------|:------|:-------|
| Optimizer | AdamW | Better weight decay handling vs Adam |
| Learning Rate | `1e-3` | Standard starting point for fine-tuning |
| Batch Size | `64` | Balanced GPU utilisation vs memory |
| Epochs | `5` | Time-constrained; best model saved by metric |
| Subset Size | `25,000` | Stratified sample to fit GPU time limit |
| Train/Val Split | `85% / 15%` | Stratified to preserve class proportions |
| Loss Function | `CrossEntropyLoss(weight=class_weights)` | Handles class imbalance |
| Mixed Precision | `torch.cuda.amp` | ~2× speedup on T4 GPU |

---

### ⚖️ Handling Class Imbalance

Inverse-frequency class weights were computed and passed to the loss function:

$$w_c = \frac{1}{N_c}, \quad \text{normalised so} \quad \bar{w} = 1$$

This forces the model to pay more attention to under-represented disease categories.

---

## 3. Experiments and Results

### 🧪 Experiments Performed

| # | Experiment | Description |
|:-:|:-----------|:------------|
| 1 | Baseline | ResNet-18, 5 epochs, full default settings |
| 2 | Subset sampling | 25k stratified vs full 51k (runtime constraint) |
| 3 | Class weighting | With vs without inverse-frequency weights in loss |
| 4 | Mixed precision | AMP enabled for faster training on T4 |

---

### 📊 Training Results (per epoch)

| Epoch | Train Loss | Val Score | Best? |
|:-----:|----------:|----------:|:-----:|
| 1 | 3.0807 | **-7.047** | ✅ Best |
| 2 | 2.9361 | -10.614 | — |
| 3 | 2.8966 | -12.495 | — |
| 4 | 2.9715 | -12.965 | — |

> 📌 Best validation score: **-7.047** (Epoch 1 checkpoint used for submission)

---

### 🏆 Competition Leaderboard Score

| Model | Public Score |
|:------|------------:|
| ResNet-18 (Epoch 1 checkpoint) | **-6.026** |

---

### 🔎 Key Observations

1. **Score degraded after Epoch 1** — the validation metric worsened even as training loss improved. This indicates the model overfits to cross-entropy loss rather than optimising the custom FN-penalising metric.

2. **Metric-loss mismatch** — `CrossEntropyLoss` is a proxy for the competition metric. The FN penalty of 5× is not captured directly by CE loss.

3. **Small subset limits generalisation** — training on 25k of 51k samples reduces potential performance, but was necessary due to Kaggle GPU time constraints.

---

## 4. Conclusion

### ✅ Summary

A ResNet-18 model fine-tuned with transfer learning was trained on a stratified 25k subset of chest X-ray images across 20 disease categories. The pipeline incorporated:

- ImageNet pre-trained feature extraction
- Inverse-frequency class weighting to combat imbalance
- Mixed-precision training for efficiency
- Stratified train/val splits for reliable evaluation

**Public Kaggle Score: `-6.026`**

---

### 📚 Key Learnings

| Learning | Implication |
|:---------|:------------|
| Metric-loss alignment matters | A custom metric-aware loss (e.g., focal loss) would improve results |
| FN penalty is severe | Prioritise recall-boosting techniques (lower thresholds, ensemble) |
| Validation score peaked early | Learning rate warmup + cosine schedule needed |
| Subset training underperforms | Full dataset + more epochs → better generalisation |

---

### 🔭 What I Would Do Next

- [ ] Train on the full 51k dataset
- [ ] Use EfficientNet-B3 or ResNet-50 for stronger features
- [ ] Add cosine learning rate schedule with warmup
- [ ] Implement test-time augmentation (TTA)
- [ ] Use a metric-aware loss (e.g., weighted focal loss tuned for FN penalty)
- [ ] Try ensemble of multiple checkpoints

---

## 5. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
2. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019.
3. PyTorch Docs — `torchvision.models`: https://pytorch.org/vision/stable/models.html
4. Kaggle Competition — `26-T-1-DL-GenAI-NPPE-1`: https://www.kaggle.com/
5. Micikevicius et al. (2018). *Mixed Precision Training.* ICLR 2018.

---

<div align="center">

*Submitted by Rajeev Patel · Roll No: 23f3002112*  
*DLGenAI NPPE 1*

</div>
