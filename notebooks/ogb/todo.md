# 🧭 Shared ML Core (Weeks 1–4) — Progress Checklist

**Goal:** build a reproducible ML pipeline on **OGBN-ArXiv** — from EDA to a first deep-learning model.

---

## 🧰 Environment & Data
- [x] Load **OGBN-ArXiv** with the official train/val/test split (no leakage)
- [x] Verify reproducibility (fixed seeds, configs, README)

---

## 🔍 Exploratory Data Analysis (EDA)
- [x] Plot target distribution across 40 classes  
- [x] Add a short takeaway on class imbalance
- [x] Visualize 128-d node features (PCA/UMAP)  
- [x] Add color by label and 1–2 sentences interpreting clusters
- [x] Create graph-structure plots (degree histogram, connectedness)
- [x] Write 3–5 bullet insights summarizing what the EDA shows

---

## 🧱 Baseline Model (Classical ML)
- [X] Implement **Multinomial Logistic Regression** on 128-d features  
- [X] Perform a small hyperparameter scan (`C`, `max_iter`)
- [X] Report **Accuracy + Macro-F1** (train/val/test)
- [ ] Add **Learning Curve** (accuracy/F1 vs. training size or epoch)
- [ ] Summarize **generalization gap** (train vs validation)

---

## ⚙️ Deep-Learning Model
- [x] Train a **Tiny MLP** on 128-d features
- [ ] Explore **one training knob** (optimizer, scheduler, dropout, normalization)
- [ ] Compare MLP vs. baseline results side-by-side

---

## 🧩 Analysis & Reflection
- [ ] Write **3–5 short error notes** (common failure patterns, rare classes, etc.)
- [x] Conduct a single **test-set probe** after freezing the final model
- [ ] Combine final metrics into a small summary table

---

### ✅ Status Summary
- **6 / 15 Complete**  
- **6 Partial (analytical additions)**  
- **3 Missing (LR baseline, learning curves, error notes)**

---

## 🎯 Next Focus
1. Add **Multinomial Logistic Regression baseline**  
2. Generate **learning curves** to visualize generalization  
3. Write **error analysis bullets** linking model mistakes to EDA insights  
4. Optionally test one **training knob** for your MLP

---

