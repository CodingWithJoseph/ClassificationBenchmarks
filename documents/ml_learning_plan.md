# Detailed Project Plan: Machine Learning & Deap Learning

## Overview & Goals

**What you'll accomplish:** Build your first machine learning pipeline from scratch, starting with data exploration and ending with a deep learning model.

**By Week 4, you'll be able to:**
- Set up a professional ML development environment
- Work with real academic research data (OGBN-ArXiv dataset)
- Build and evaluate both traditional and deep learning models
- Understand how well your models actually work (avoiding overfitting)
- Create reproducible experiments

**Prerequisites:** Basic Python knowledge only - no ML experience needed!

---

## Tools & Setup

### Required Software
- **Python 3.10+**: Programming language
- **Conda or virtualenv**: Creates isolated environments to avoid package conflicts
- **PyTorch**: Deep learning framework (like TensorFlow but more beginner-friendly)
- **scikit-learn**: Traditional ML algorithms library
- **matplotlib/plotly**: Data visualization libraries

### Optional (Advanced)
- **Hugging Face Transformers**: Pre-trained language models (Week 3-4)
- **OGB + PyTorch Geometric**: Specialized graph neural network tools

### Setup Checklist
- [X] Install Python 3.10+
- [X] Create virtual environment
- [X] Install packages:
- [X] Test installation with simple import statements

---

## Dataset: OGBN-ArXiv Explained

**What it is:** Academic paper abstracts from ArXiv.org with 40 subject categories (like "Machine Learning," "Physics," etc.)

**Key features:**
- **169,343 papers** (nodes in a graph)
- **128-dimensional features** per paper (numerical representations of the text)
- **40 classes** (subject categories to predict)
- **Time-based split**: Older papers for training, newer for testing (realistic scenario)

**Why this dataset:** It's challenging enough to be interesting but small enough to run on a laptop.

---

## Important Rules (Guardrails)

### 🚨 Critical Rules - Don't Break These!
1. **Use official data splits only** - Don't create your own train/test splits
2. **Keep test set sacred** - Only look at test results once at the very end
3. **Always compare train vs validation** - This shows if your model is overfitting
4. **Set random seeds** - Makes your experiments reproducible
5. **Document everything** - Save configs, write a simple README

### Evaluation Metrics Explained
- **Accuracy**: Percentage of correct predictions (simple but can be misleading with imbalanced data)
- **Macro-F1**: Average F1 score across all classes (better for imbalanced data)
- **F1 Score**: Harmonic mean of precision and recall (balances false positives and false negatives)

---

## Week 1-2: Data Explorer Phase

### Goal: Understand your data before building models

### Week 1 Actionable Tasks
- [X] Task 1: Environment Setup (Day 1)
- [X] Task 2: Load and Examine Data (Days 1-2)

**Deliverable:** Working dataset loading script

### Week 2 Actionable Tasks

Task 1: Exploratory Data Analysis (EDA) - Create these 3 visualizations:
- [X] Target Distribution Bar Chart
- [ ] Feature Visualization (2D)

**Deliverable:** 3-4 saved plots with 3-5 key insights written down

#### Task 2: Baseline Model (Logistic Regression or XGBoost)
- **Key Concept - Generalization Gap:** The difference between training and validation performance. If training accuracy is 90% but validation is 70%, you have a 20% generalization gap (overfitting).

**Deliverable:** Baseline model with documented train vs validation metrics

---

## Week 3-4: Deep Learning Transition

### Goal: Build your first neural network and understand training dynamics

### Week 3 Actionable Tasks
- [ ] Task 1: Hyperparameter Search on Baseline
Test different settings to find the best baseline:

**Deliverable:** Optimized baseline model with best hyperparameters

- [ ] Task 2: First Neural Network (Simple MLP)

**MLP Definition:** Multi-Layer Perceptron - a neural network with input layer, hidden layers, and output layer.

**Key Terms:**
- **Hidden dimension**: Number of neurons in hidden layer (64 is small, 256+ is larger)
- **ReLU**: Activation function that helps networks learn complex patterns
- **Dropout**: Randomly sets some neurons to zero during training to prevent overfitting
- **Adam optimizer**: Smart way to update model weights during training

### Week 4 Actionable Tasks

- [ ] Task 1: Training Loop with Monitoring
- [ ] Task 2: Experiment with Training "Knobs"
Test these one at a time and compare results:

**Learning Rate Schedulers:**
**Different Optimizers:**
**Normalization Layers:**

**Deliverable:** Comparison table showing how each change affects validation performance

### Week 4 Success Checklist
- [ ] Optimized baseline model locked in
- [ ] Working neural network that trains successfully
- [ ] At least one training improvement tested and documented
- [ ] Learning curves plotted (loss going down over time)
- [ ] Final train vs validation comparison completed
- [ ] Optional: One final test on held-out test set

---

## Beyond Week 4: Advanced Tracks

### NLP Track: Text Processing
**Goal:** Work directly with the paper abstracts (text) instead of pre-computed features

**Progression:**
1. **TF-IDF**: Count word frequencies (traditional approach)
2. **BERT embeddings**: Use pre-trained language model (modern approach)
3. **Compare**: Which works better and why?

### Graph Track: Neural Networks on Graphs
**Goal:** Use the citation network structure (which papers cite which)

**Progression:**
1. **Graph Convolutional Network (GCN)**: Neural network that uses graph structure
2. **GraphSAGE**: More advanced graph neural network
3. **Compare**: Does graph structure help vs just using features?

**Final Success Metric:** You should be able to explain to someone else what your model does, how well it works, and what you learned about the data!