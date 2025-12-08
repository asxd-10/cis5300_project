# Strong Baseline + Extended Model: BiLSTM-CRF for Scientific Section Classification.

We implement a **BiLSTM-CRF model** for sentence-level section classification on the **PubMed 200K RCT dataset**, following and extending the architecture originally used in Dernoncourt & Lee (2017). The Strong baseline uses a 1-layer BiLSTM, while our extended model increases embedding capacity, introduces regularization, deepens the recurrent stack, and applies improved optimization strategies. These enhancements result in **substantial performance gains** across all metrics.

---

## 1. Strong Baseline Model Architecture

### **BiLSTM-CRF**
The model consists of:

- **Embedding Layer**  
  Converts token IDs into dense vectors (100-dim).
- **Bidirectional LSTM**  
  Hidden size: 128 per direction → 256 combined.
- **Linear Classifier**  
  Maps BiLSTM outputs to tag logits.
- **CRF Layer**  
  Captures dependency between adjacent labels.

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
- Increased embedding dimension **100 → 300** for richer token semantics  
- Added dropout **0.35** on both embedding output and LSTM output  
- Increased LSTM depth **1 → 2 layers**  
- Added **learning-rate scheduler (ReduceLROnPlateau)** to escape plateaus  
- Enabled **gradient clipping (max_norm=1.0)** to avoid exploding gradients  
- Added **early stopping** based on dev Macro-F1 (patience=3)  
- Trained up to **15 epochs**, but early stopping usually triggered around epoch 10  
- Final hidden size = **256** (BiLSTM 128×2)  

---
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
| **Embedding Dim** | 100 | **300** |
| **LSTM Layers** | 1 | **2** |
| **LSTM Hidden Dim** | 128 | **128 (256 bidirectional)** |
| **Dropout** | 0 | **0.35 (embedding + LSTM)** |
| **Extra Dropout Layer** | No | **Yes** |
| **Optimizer** | Adam | Adam |
| **Learning Rate** | 0.001 fixed | **0.001 + ReduceLROnPlateau** |
| **LR Scheduler** | No | **Yes (factor=0.5, patience=3)** |
| **Gradient Clipping** | No | **Yes (max_norm=1.0)** |
| **Early Stopping** | No | **Yes (patience=3)** |
| **Batch Size** | 32 | 32 |
| **Epochs** | 3–5 | **Up to 15 (early stop ~10)** |

---

**Loss Function** : We minimize the negative log-likelihood from the CRF.  

## **Evaluation Metrics**

We computed:
- Accuracy
- Precision
- Recall
- F1-score (macro + weighted)
- Per-label detailed report



## 4. Baseline Performance

| Metric | Dev | Test |
|--------|------|------|
| Macro-F1 | 0.8381 | 0.8346 |
| Accuracy | 0.8943 | 0.8885 |

---

## 5. Improved Model Performance

### Development Set
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| BACKGROUND | 0.7759 | 0.8049 | 0.7901 | 3449 |
| OBJECTIVE | 0.7272 | 0.6831 | 0.7044 | 2376 |
| METHODS | 0.9530 | 0.9667 | 0.9598 | 9964 |
| RESULTS | 0.9522 | 0.9525 | 0.9524 | 9841 |
| CONCLUSIONS | 0.9600 | 0.9323 | 0.9460 | 4582 |
| **Accuracy** | | | **0.9161** | 30212 |
| **Macro Avg** | 0.8736 | 0.8679 | **0.8705** | 30212 |
| **Weighted Avg**| 0.9158 | 0.9161 | 0.9158 | 30212 |

### Test Set
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| BACKGROUND | 0.7811 | 0.8031 | 0.7919 | 3621 |
| OBJECTIVE | 0.7081 | 0.6622 | 0.6844 | 2333 |
| METHODS | 0.9433 | 0.9683 | 0.9556 | 9897 |
| RESULTS | 0.9554 | 0.9453 | 0.9503 | 9713 |
| CONCLUSIONS | 0.9632 | 0.9398 | 0.9514 | 4571 |
| **Accuracy** | | | **0.9130** | 30135 |
| **Macro Avg** | 0.8702 | 0.8638 | **0.8667** | 30135 |
| **Weighted Avg**| 0.9125 | 0.9130 | 0.9126 | 30135 |

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

## How to run it:

To run it:

1. Upload/open the notebook in **Google Colab**  
2. Go to **Runtime → Change runtime type → GPU**  
3. Run all cells sequentially  
4. The notebook performs:
   - Data loading  
   - Preprocessing  
   - Model construction  
   - Training  
   - Saving model  
   - Evaluation (dev + test sets)

No `.py` files are used, everything is self-contained in the notebook.

Run on Google Colab with GPU acceleration.
