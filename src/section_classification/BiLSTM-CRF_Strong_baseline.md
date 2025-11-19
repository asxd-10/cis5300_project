# **Strong Baseline: BiLSTM-CRF for Section Classification**

This document describes the **strong baseline model** used for scientific abstract **section classification**.  
We implement a **BiLSTM-CRF** architecture on the **PubMed 200K RCT dataset**, a widely used and competitive sequence-labeling model.  

Dernoncourt & Lee (2017), "PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts"

The key idea:

Use BiLSTM to model sequential context across sentences in each abstract.

Use a CRF layer to model label dependencies (e.g., METHODS → RESULTS → CONCLUSIONS).

This baseline significantly outperforms simple approaches such as majority class.
---

## **1. Model Architecture**

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

### **Model Printout**
```
SentenceBiLSTM_CRF(
  (embedding): Embedding(69734, 100, padding_idx=0)
  (lstm): LSTM(100, 128, batch_first=True, bidirectional=True)
  (hidden2tag): Linear(in_features=256, out_features=5, bias=True)
  (crf): CRF(num_tags=5)
)
```


---

## **2. Data Setup**

We use:
- **Train set**
- **Dev set**
- **Test set** (only for final reporting)

Padding mask is computed using a boolean attention mask.

We use:
- `PAD_ID = 0`
- `MAX_LEN = 512`

---

## **3. Training Details**

### **Hyperparameters**
| Parameter | Value |
|----------|-------|
| Embedding dim | 100 |
| LSTM hidden size | 128 |
| Layers | 1 |
| Optimizer | Adam |
| LR | 0.001 |
| Batch size | 32 |
| Epochs | 3–5 |
| CRF reduction | sum |

### **Loss Function**
We minimize the **negative log likelihood** from the CRF layer.

### **Model Saving**
At the end of training:
```python
torch.save(model.state_dict(), "best_bilstm_crf_model.pt")
```

To load later:

```python
model.load_state_dict(torch.load("best_bilstm_crf_model.pt"))
model.eval()
```

## **4.Evaluation Metrics**

We computed:
- Accuracy
- Precision
- Recall
- F1-score (macro + weighted)
- Per-label detailed report


## **5. Baseline Results**

Detailed Classification Report for Test set

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| BACKGROUND       | 0.7504    | 0.6943 | 0.7213   | 3621    |
| OBJECTIVE        | 0.6064    | 0.6730 | 0.6380   | 2333    |
| METHODS          | 0.9265    | 0.9543 | 0.9402   | 9897    |
| RESULTS          | 0.9438    | 0.9289 | 0.9363   | 9713    |
| CONCLUSIONS      | 0.9509    | 0.9243 | 0.9374   | 4571    |
| **Accuracy**     | -         | -      | 0.8885   | 30135   |
| **Macro Avg**    | 0.8356    | 0.8349 | 0.8346   | 30135   |
| **Weighted Avg** | 0.8899    | 0.8885 | 0.8888   | 30135   |


Detailed Classification Report for Dev Set

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| BACKGROUND       | 0.7321    | 0.6918 | 0.7114   | 3449    |
| OBJECTIVE        | 0.6267    | 0.6768 | 0.6507   | 2376    |
| METHODS          | 0.9402    | 0.9561 | 0.9481   | 9964    |
| RESULTS          | 0.9443    | 0.9435 | 0.9439   | 9841    |
| CONCLUSIONS      | 0.9532    | 0.9197 | 0.9361   | 4582    |
| **Accuracy**     | -         | -      | 0.8943   | 30212   |
| **Macro Avg**    | 0.8393    | 0.8376 | 0.8381   | 30212   |
| **Weighted Avg** | 0.8951    | 0.8943 | 0.8945   | 30212   |


Strong baseline macro-F1:
- Test: 0.8346
- Dev: 0.8381


## **Notebook Location**

The full implementation is available here:
` cis5300_project/notebooks/BILSTM_section_classification.ipynb`

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
