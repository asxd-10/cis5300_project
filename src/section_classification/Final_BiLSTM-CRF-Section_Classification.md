# Strong Baseline + Extended Model: BiLSTM-CRF for Scientific Section Classification.

We implement a **BiLSTM-CRF model** for sentence-level section classification on the **PubMed 200K RCT dataset**, following and extending the architecture originally used in Dernoncourt & Lee (2017). The baseline uses a 1-layer BiLSTM, while our extended model increases embedding capacity, introduces regularization, deepens the recurrent stack, and applies improved optimization strategies. These enhancements result in **substantial performance gains** across all metrics.

---

## 1. Baseline Model Architecture

The original baseline uses the following components:

* **Embedding Layer:** 100-dim embeddings
* **BiLSTM Layer:** 1 layer, hidden size = 128 per direction
* **Linear Classifier:** maps 256-dim features to 5 labels
* **CRF Layer:** models global label dependencies

Baseline model printout:

```python
SentenceBiLSTM_CRF(
  (embedding): Embedding(69734, 100, padding_idx=0)
  (lstm): LSTM(100, 128, batch_first=True, bidirectional=True)
  (hidden2tag): Linear(in_features=256, out_features=5, bias=True)
  (crf): CRF(num_tags=5)
)
```

## 2. Extended / Improved Model Architecture

### Improvements Made
- Embedding dimension **100 → 300**
- Dropout **0.35** added on embeddings + LSTM outputs
- LSTM layers **1 → 2**
- Learning-rate scheduler: **ReduceLROnPlateau**
- Gradient clipping enabled
- Early stopping on **dev macro-F1**

**Improved model printout:**

```python
SentenceBiLSTM_CRF(
(embedding): Embedding(69734, 300, padding_idx=0)
(dropout): Dropout(p=0.35, inplace=False)
(lstm): LSTM(300, 128, num_layers=2, batch_first=True, dropout=0.35, bidirectional=True)
(hidden2tag): Linear(in_features=256, out_features=5, bias=True)
(crf): CRF(num_tags=5)
)
```

## 3. Training Setup

### Hyperparameters
| Parameter | Baseline | Improved |
|----------|----------|----------|
| Embedding Dim | 100 | **300** |
| LSTM Layers | 1 | **2** |
| LSTM Dropout | 0 | **0.35** |
| Extra Dropout Layer | No | **Yes** |
| LR Scheduler | No | **Yes** |
| Gradient Clipping | No | **Yes** |
| Early Stopping | No | **Yes** |
| Hidden Size | 128 | 128 |
| Optimizer | Adam | Adam |
| Batch Size | 32 | 32 |
| Epochs | 3–5 | Up to 10 |

---

**Loss Function** : We minimize the negative log-likelihood from the CRF.  


## 4. Baseline Performance

| Metric | Dev | Test |
|--------|------|------|
| Macro-F1 | 0.8381 | 0.8346 |
| Accuracy | 0.8943 | 0.8885 |

---

## 5. Improved Model Performance

### Development Set
| Class | Precision | Recall | F1 | Support |
|--------|-----------|--------|------|---------|
| BACKGROUND | 0.7759 | 0.8049 | 0.7901 | 3449 |
| OBJECTIVE | 0.7272 | 0.6831 | 0.7044 | 2376 |
| METHODS | 0.9530 | 0.9667 | 0.9598 | 9964 |
| RESULTS | 0.9522 | 0.9525 | 0.9524 | 9841 |
| CONCLUSIONS | 0.9600 | 0.9323 | 0.9460 | 4582 |
| **Accuracy** | **0.9161** | |
| **Macro-F1** | **0.8705** | |

### Test Set
| Class | Precision | Recall | F1 | Support |
|--------|-----------|--------|------|---------|
| BACKGROUND | 0.7811 | 0.8031 | 0.7919 | 3621 |
| OBJECTIVE | 0.7081 | 0.6622 | 0.6844 | 2333 |
| METHODS | 0.9433 | 0.9683 | 0.9556 | 9897 |
| RESULTS | 0.9554 | 0.9453 | 0.9503 | 9713 |
| CONCLUSIONS | 0.9632 | 0.9398 | 0.9514 | 4571 |
| **Accuracy** | **0.9130** |
| **Macro-F1** | **0.8667** |

---

## 6. Performance Comparison

| Metric | Baseline | Improved | Δ Improvement |
|--------|----------|----------|----------------|
| Test Macro-F1 | 0.8346 | **0.8667** | **+3.2** |
| Dev Macro-F1 | 0.8381 | **0.8705** | **+3.4** |
| Test Accuracy | 0.8885 | **0.9130** | **+2.5** |

---

## 7. Error Analysis

### Confusion Matrix
<img width="658" height="547" alt="image" src="https://github.com/user-attachments/assets/7e054a16-fd63-4f7f-be29-be132cb74357" />


### Sentence-Length Distribution of Errors  
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/0bd38d4d-8f18-491f-85a2-f5c968a15de5" />


### Error Statistics
- **Total sentences:** 30,135  
- **Total misclassified:** 2,621  
- **Error rate:** **8.70%**

### Errors per True Class
| Label | Class | Errors |
|-------|--------|---------|
| 1 | OBJECTIVE | **788** |
| 0 | BACKGROUND | **713** |
| 2 | METHODS | **314** |
| 3 | RESULTS | **531** |
| 4 | CONCLUSIONS | **275** |

### Sample Misclassifications
1. True: 3, Pred: 2 — “a post hoc analysis was conducted with the use …”  
2. True: 3, Pred: 2 — “liver function tests were measured at …”  
3. True: 3, Pred: 4 — “nor was evar superior regarding cost-utility …”  
4. True: 0, Pred: 1 — “evidence suggests that individuals with social anxiety demonstrate vigilance …”  
5. True: 2, Pred: 1 — “this study investigated whether oxytocin can affect attentional bias …”

---

## 8. Notebook
Implementation lives in: ```cis5300_project/notebooks/Final_Extension_BILSTM_section_classification.ipynb```   

Run on Google Colab with GPU acceleration.
