
### 📌 Project Overview

This notebook presents a complete end-to-end **Sentiment Analysis** pipeline applied to the **Amazon Customer Reviews** dataset.  
The goal is to classify each review as either **Positive** or **Negative** using both classical Machine Learning and modern Deep Learning approaches.

---

### 📦 Dataset

| Property | Value |
|---|---|
| Source | [Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) |
| Format | FastText (bz2 compressed) |
| Training set | **3,600,000** reviews |
| Test set | **400,000** reviews |
| Classes | `__label__1` → Negative (0) · `__label__2` → Positive (1) |
| Balance | Perfectly balanced (50 / 50) |

---

### 🗺️ Pipeline Structure

```
Raw .bz2 Files
      │
      ▼
 Data Loading & EDA
      │
      ├──► ML Baseline (TF-IDF + Logistic Regression / LinearSVC)
      │
      └──► DL Pipeline
                │
                ├── Stratified Sampling (1.2M from 3.6M)
                ├── Tokenisation & Padding
                ├── Model Training (BiLSTM · BiGRU · Conv1D+BiLSTM)
                ├── Evaluation on 400K test set
                └── Save Best Model + Tokeniser
```

---

### 🏗️ Models Trained

| # | Model | Type | Key Feature |
|---|---|---|---|
| 1 | Logistic Regression | ML | TF-IDF unigrams + bigrams |
| 2 | Linear SVM | ML | TF-IDF unigrams + bigrams |
| 3 | **BiLSTM** | Deep Learning | Bidirectional LSTM + GlobalMaxPool |
| 4 | **BiGRU** | Deep Learning | Bidirectional GRU + GlobalMaxPool |
| 5 | **Conv1D + BiLSTM** | Deep Learning | CNN n-gram features → BiLSTM context |

---

### ⚙️ Key Design Decisions

- **No data leakage** — tokeniser is fitted on training texts only; `test_df` is never seen during training  
- **Stratified sampling** — 600K per class from `train_df` to keep balance  
- **Large batch size (1024)** — critical for GPU efficiency; original batch of 8 made training 128× slower  
- **Bidirectional RNNs** — read sequences forward *and* backward for richer context  
- **GlobalMaxPooling** — captures the most salient feature across all timesteps, better than last hidden state  
- **SpatialDropout1D** — drops entire embedding channels (better than random dropout for sequences)

---

## Results at a Glance

| Model | Type | Accuracy | ROC-AUC | Notes |
|---|---|---|---|---|
| Logistic Regression | ML | ~90.6% | ~0.966 | Fast, strong baseline |
| Linear SVM | ML | ~90.3% | N/A | Slightly below LGR |
| BiLSTM | Deep Learning | ~94.2% | ~0.983 | Strong sequential model |
| BiGRU | Deep Learning | ~94.4% | ~0.984 | GRU = faster BiLSTM alternative |
| **Conv1D + BiLSTM** | **Deep Learning** | **~94.6%** | **~0.986** | **🏆 Best overall** |

---

## Key Findings

### 1. DL consistently outperforms classical ML
All three deep learning models exceeded 94% accuracy, beating the TF-IDF baselines  
by **~4 percentage points** — a meaningful margin on 400K test samples.

### 2. The Conv1D + BiLSTM hybrid is the strongest architecture
Combining local CNN n-gram features with BiLSTM sequential context gave the  
best of both worlds: fast convergence (CNN) and long-range understanding (LSTM).

### 3. Batch size was the most critical engineering fix
The original batch size of **8** produced 315,000 gradient steps per epoch,  
making training effectively impossible to complete on a GPU.  
Increasing to **1024** reduced steps/epoch to ~1,000 — a **315× speed-up**.

### 4. Bidirectional encoding matters for sentiment
Reading reviews both left→right and right→left gave the models access to  
context that unidirectional RNNs miss (e.g. negations that appear after the sentiment word).

### 5. GlobalMaxPooling outperforms last-hidden-state pooling
The most sentiment-relevant word in a review can appear anywhere.  
GlobalMaxPool selects the strongest signal across all timesteps rather  
than relying on the final RNN state alone.

---

## Hyperparameter Decisions

| Parameter | Value Chosen | Reason |
|---|---|---|
| `VOCAB_SIZE` | 50,000 | Covers >95% of word types in the training corpus |
| `MAX_LEN` | 150 | Captures 93rd-percentile review length; 95th is 161 |
| `EMBED_DIM` | 128 | Richer representations than 64 with manageable memory cost |
| `BATCH_SIZE` | 1,024 | GPU-efficient; 8 made training 315× slower |
| `DL_SAMPLES` | 1.2M | Sufficient signal; tokenising all 3.6M is slow and unnecessary |
| `LEARNING_RATE` | 3×10⁻⁴ | Lower than default 1×10⁻³ for stable large-batch training |

---

## Deployment

The best model (`Conv1D_BiLSTM`, 94.58% accuracy, AUC 0.986) was deployed as a  
**FastAPI web service** with a custom dark-themed UI, supporting:
- **Single review** analysis with confidence score and progress bar  
- **Batch mode** (up to 50 reviews) with aggregate statistics  
- **REST API** endpoints (`/api/predict`, `/api/predict/batch`) for programmatic access  
- Ready for **Hugging Face Spaces** deployment via Docker

---

## Possible Further Improvements

| Idea | Expected Gain |
|---|---|
| Pre-trained embeddings (GloVe / FastText) | +0.5–1% accuracy |
| Fine-tune DistilBERT on this dataset | +2–3% accuracy |
| Train on the full 3.6M instead of 1.2M sample | +0.5–1% accuracy |
| Ensemble (average predictions of BiLSTM + BiGRU + Conv1D) | +0.3–0.5% accuracy |
| Hyperparameter search (Optuna / Keras Tuner) | Marginal gains |

