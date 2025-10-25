
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
- [x] **Implement baseline TF-IDF encoding**
  - Use `sklearn.feature_extraction.text.TfidfVectorizer`
  - Set parameters: `max_features=10000`, `stop_words='english'`
  - Train logistic regression classifier, record accuracy & F1-macro

- [x] **Test TF-IDF with n-grams**
  - Experiment with `ngram_range=(1,2)`, `(1,3)`, and `(1,5)`
  - Compare performance and feature dimensionality
  - Analyze which n-gram setting works best and why

- [x] **Apply traditional NLP preprocessing**
  - Test with/without: stopword removal, lemmatization, stemming
  - Use `nltk` or `spacy` for text preprocessing
  - Document which preprocessing steps improve performance

- [x] **Analyze TF-IDF results**
  - Record all combinations in results table
  - Explain performance differences (vocabulary size, sparsity, etc.)

### Word2Vec Experiments:
- [x] **Use pretrained Word2Vec model**
  - Load `word2vec-google-news-300` from `gensim`
  - Average word vectors for each document (handle OOV words)
  - Train logistic regression on 300-d embeddings

- [x] **Record Word2Vec performance**
  - Compare against TF-IDF results
  - Analyze semantic vs. syntactic representation differences

- [x] **[Optional] Train custom Word2Vec**
  - Train Word2Vec on ArXiv training set abstracts
  - Use `gensim.models.Word2Vec` with `vector_size=300`, `window=5`
  - Compare custom vs. pretrained model performance

### Modern Embedding Methods:
- [x] **Implement sentence-level embeddings**
  - Use models like `sentence-transformers/all-MiniLM-L6-v2`
  - Or API-based: OpenAI embeddings, Google embeddings
  - Generate document-level embeddings

- [x] **Test modern embedding performance**
  - Apply same logistic regression evaluation
  - Record performance in comparison table

- [x] **Create comprehensive comparison**
  - Build results table with all encoding methods
  - Include: accuracy, F1-macro, embedding dimension, training time
  - Analyze which methods work best for scientific text classification

---

## 🤖 Weeks 8-12: Modern Transformer Methods
**Focus:** Implement modern transformer architectures from scratch to fine-tuned models.

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