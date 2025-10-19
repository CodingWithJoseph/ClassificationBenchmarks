
# 🧭 NLP & Transformer Pipeline (Weeks 5–12) — Progress Checklist

**Goal:** Build comprehensive text-based models on **OGBN-ArXiv** — from data mapping to modern transformer architectures.

---

## 🔗 Week 5: Mapping & Sanity Check
**Focus:** Map the text to the node, and make sure the mapping is legitimate.

- [x] **Load OGBN-ArXiv node mapping data**
  - Load the `nodeidx2paperid.csv.gz` file from the dataset
  - Extract `node_index` → `paper_id` relationships

- [x] **Acquire ArXiv text data**
  - Download or access ArXiv metadata containing titles and abstracts
  - Ensure you have `paper_id`, `title`, `abstract`, `year` columns

- [x] **Build deterministic join pipeline**
  - Create `node_index` → `paper_id` → `title/abstract` mapping
  - Handle missing entries and document coverage statistics

- [x] **Canonicalize paper IDs**
  - Strip version suffixes (e.g., remove `v1`, `v2`, `v3` from `1234.5678v2`)
  - Convert to lowercase and trim whitespace
  - Normalize Unicode characters for consistent matching

- [x] **Create unified text dataset**
  - Combine `title + " " + abstract` into single `text` column
  - Generate `arxiv_text.parquet` with columns: `node_index`, `paper_id`, `year`, `text`, `label`, `split`

- [x] **Validate label mapping sanity check**
  - Load label category mapping from `labelidx2arxivcategory.csv.gz`
  - Sample 10-20 random entries and manually verify text content matches assigned category
  - Document any obvious mismatches or data quality issues

---

## 🔤 Weeks 6-8: Encoding Exploration
**Focus:** Explore different text encoding methods with linear probes and compare performance.

### TF-IDF Experiments:
- [ ] **Implement baseline TF-IDF encoding**
  - Use `sklearn.feature_extraction.text.TfidfVectorizer`
  - Set parameters: `max_features=10000`, `stop_words='english'`
  - Train logistic regression classifier, record accuracy & F1-macro

- [ ] **Test TF-IDF with n-grams**
  - Experiment with `ngram_range=(1,2)`, `(1,3)`, and `(1,5)`
  - Compare performance and feature dimensionality
  - Analyze which n-gram setting works best and why

- [ ] **Apply traditional NLP preprocessing**
  - Test with/without: stopword removal, lemmatization, stemming
  - Use `nltk` or `spacy` for text preprocessing
  - Document which preprocessing steps improve performance

- [ ] **Analyze TF-IDF results**
  - Record all combinations in results table
  - Explain performance differences (vocabulary size, sparsity, etc.)

### Word2Vec Experiments:
- [ ] **Use pretrained Word2Vec model**
  - Load `word2vec-google-news-300` from `gensim`
  - Average word vectors for each document (handle OOV words)
  - Train logistic regression on 300-d embeddings

- [ ] **Record Word2Vec performance**
  - Compare against TF-IDF results
  - Analyze semantic vs. syntactic representation differences

- [ ] **[Optional] Train custom Word2Vec**
  - Train Word2Vec on ArXiv training set abstracts
  - Use `gensim.models.Word2Vec` with `vector_size=300`, `window=5`
  - Compare custom vs. pretrained model performance

### Modern Embedding Methods:
- [ ] **Implement sentence-level embeddings**
  - Use models like `sentence-transformers/all-MiniLM-L6-v2`
  - Or API-based: OpenAI embeddings, Google embeddings
  - Generate document-level embeddings

- [ ] **Test modern embedding performance**
  - Apply same logistic regression evaluation
  - Record performance in comparison table

- [ ] **Create comprehensive comparison**
  - Build results table with all encoding methods
  - Include: accuracy, F1-macro, embedding dimension, training time
  - Analyze which methods work best for scientific text classification

---

## 🤖 Weeks 8-12: Modern Transformer Methods
**Focus:** Implement modern transformer architectures from scratch to fine-tuned models.

### Encoder-Only Transformer From Scratch:
- [ ] **Design small transformer architecture**
  - Define model: 2-4 layers, 4-8 heads, 256-512 hidden dimension
  - Use `torch.nn.TransformerEncoder` or implement custom blocks
  - Add classification head for 40 classes

- [ ] **Configure training pipeline**
  - Set up tokenizer (e.g., `AutoTokenizer.from_pretrained('bert-base-uncased')`)
  - Define optimizer: Adam with lr=1e-4, weight decay
  - Use learning rate scheduler and early stopping

- [ ] **Train and evaluate custom transformer**
  - Train for 10-20 epochs with validation monitoring
  - Record: validation accuracy, F1-macro, training curves
  - Document overfitting patterns and training stability

### BERT-based Experiments:
- [ ] **Implement frozen BERT classifier**
  - Load `bert-base-uncased` from `transformers`
  - Freeze encoder weights, train only classification head
  - Use 512 token limit, handle longer texts with truncation

- [ ] **Test BERT fine-tuning (if VRAM allows)**
  - Unfreeze last 2-4 layers or full model
  - Use lower learning rate (1e-5) for pretrained layers
  - Monitor for catastrophic forgetting

- [ ] **[Optional] Implement LoRA fine-tuning**
  - Use `peft` library for Low-Rank Adaptation
  - Set `r=16`, `lora_alpha=32`, target attention layers
  - Compare LoRA vs. full fine-tuning stability

### Advanced Transformer Models:
- [ ] **Evaluate DeBERTa model**
  - Use `microsoft/deberta-base` for comparison
  - Apply same frozen/fine-tuned experimental setup
  - Note disentangled attention and relative position benefits

- [ ] **Compare transformer architectures**
  - Create results table: Custom, BERT-frozen, BERT-finetuned, DeBERTa
  - Include: accuracy, F1-macro, training time, inference speed
  - Document attention pattern analysis (optional)

### Final Analysis & Documentation:
- [ ] **Compile comprehensive results table**
  - Include all methods from TF-IDF to transformer models
  - Add columns: model type, parameters, accuracy, F1-macro, training time

- [ ] **Write detailed semester report**
  - Document methodology for each approach
  - Analyze performance trends across different text encoding methods
  - Discuss computational vs. performance trade-offs
  - Provide recommendations for scientific text classification

- [ ] **Create reproducible code pipeline**
  - Ensure all experiments can be reproduced with fixed seeds
  - Document hardware requirements and training times
  - Include hyperparameter configurations and model checkpoints

---

## 📊 Expected Deliverables by Week 12:

### Data Assets:
- [ ] `arxiv_text.parquet` - Clean, mapped text dataset
- [ ] Label validation report with sanity check examples

### Model Results:
- [ ] TF-IDF experiments (3+ configurations)
- [ ] Word2Vec results (pretrained + optional custom)
- [ ] Modern embedding baselines
- [ ] Custom transformer implementation
- [ ] BERT-family model comparisons

### Documentation:
- [ ] Comprehensive performance comparison table
- [ ] Semester-long methodology report
- [ ] Code repository with reproducible experiments
- [ ] Insights on text encoding effectiveness for scientific documents

---

### ✅ Status Summary
- **0 / 35 Complete**  
- **All items pending - ready to start Week 5**

---

## 🎯 Next Focus (Week 5):
1. Set up data mapping pipeline from `node_index` → `paper_id` → `text`
2. Implement paper ID canonicalization (version stripping, normalization)
3. Create unified `arxiv_text.parquet` dataset
4. Perform label sanity checking with manual validation of 10-20 samples

---