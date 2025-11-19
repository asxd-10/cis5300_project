# Strong Baseline: SciBERT for Claim Verification

## Overview

Our strong baseline uses **SciBERT** (Beltagy et al., 2019) fine-tuned for scientific claim verification with multi-task learning:
- **Task 1**: Predict claim label (SUPPORT, CONTRADICT, NOT_ENOUGH_INFO)
- **Task 2**: Extract evidence sentences (binary classification per sentence)

**Architecture**: Two-stage pipeline with oracle retrieval + joint prediction.

## Requirements

```bash
pip install torch transformers datasets jsonlines scikit-learn tqdm
```

**Hardware**: GPU pref (Google Colab T4 works too)

## Quick Start

### Option 1: Use Pre-trained Model (Fastest)

```bash
# Download our trained checkpoint
# Generate predictions
python src/claim_verification/inference.py \
  --model models/claim_verifier/model_epoch8_final.pt \
  --split dev \
  --output output/dev/scibert_predictions.jsonl
```

### Option 2: Train from Scratch (Recommended)

```bash
# Open training notebook
jupyter notebook notebooks/train_scibert_baseline.ipynb

# Or use Google Colab:
# https://colab.research.google.com/
# Upload train_scibert_baseline.ipynb
```

## Training Details

### Model Architecture

```python
from src.claim_verification.model import ClaimVerifier

model = ClaimVerifier(
    model_name='allenai/scibert_scivocab_uncased',
    num_labels=3,           # SUPPORT, CONTRADICT, NEI
    max_sentences=20        # Max sentences per abstract
)
```

**Components**:
1. **Encoder**: SciBERT (110M parameters, pretrained on 1.14M papers)
2. **Label Classifier**: Linear layer on `[CLS]` token → 3-way classification
3. **Evidence Classifier**: Linear layer on sentence boundary tokens → binary per sentence

### Input Format

```
[CLS] claim [SEP] sentence1 [SEP] sentence2 [SEP] ... [PAD]
```

**Example**:
```
[CLS] Vitamin D reduces infections [SEP] 
Background: Vitamin D deficiency is common [SEP]
Methods: We conducted a trial [SEP]
Results: Supplementation reduced infections by 12% [SEP]
...
```

### Training Configuration

```python
# Hyperparameters (optimized through experimentation)
num_epochs = 8 (tuning from 4 to 12 - 8 works decent)
batch_size = 8 (tuning from 4 to 18)
learning_rate = 2e-5 
evidence_weight = 2.0   (tuning from 0.5 to 2.5 - more experimentation needed)       # Weight for evidence loss

# Multi-task loss
loss = label_loss + evidence_weight * evidence_loss
```

**Training Data**: 505 claims with evidence (filtered from 809 total)

**Training Time**: ~25 minutes on Google Colab T4 GPU depends on epochs

### Training Process

1. **Load Data**:
   ```python
   from src.common.data_utils import load_claims, load_corpus
   
   train_claims = load_claims('data/scifact/data/claims_train.jsonl')
   corpus = load_corpus('data/scifact/data/corpus.jsonl')
   ```

2. **Create Dataset**:
   ```python
   from torch.utils.data import Dataset, DataLoader
   
   train_dataset = SciFact_Dataset(train_claims, corpus, model.tokenizer)
   train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
   ```

3. **Training Loop**:
   ```python
   optimizer = AdamW(model.parameters(), lr=2e-5)
   criterion = nn.CrossEntropyLoss()
   
   for epoch in range(num_epochs):
       for batch in train_loader:
           # Forward pass
           label_logits, evidence_logits = model(
               input_ids, attention_mask, sentence_positions
           )
           
           # Multi-task loss
           label_loss = criterion(label_logits, labels)
           evidence_loss = BCEWithLogitsLoss()(evidence_logits, evidence_mask)
           loss = label_loss + 2.0 * evidence_loss
           
           # Backward pass
           loss.backward()
           optimizer.step()
   ```

4. **Save Checkpoint**:
   ```python
   torch.save(model.state_dict(), 'models/claim_verifier/model_epoch8.pt')
   ```

## Generating Predictions

### Using the Trained Model

```python
# Load trained model
model = ClaimVerifier()
model.load_state_dict(torch.load('models/claim_verifier/model_epoch8.pt'))
model.eval()

# Generate predictions
predictions = []
for claim in dev_claims:
    # Use oracle retrieval
    doc_id = int(claim.cited_doc_ids[0])
    doc = corpus[doc_id]
    
    # Create input
    text = claim.claim
    for sent in doc.abstract[:10]:
        text += " [SEP] " + sent
    
    # Tokenize
    encoding = tokenizer(text, max_length=512, padding='max_length', 
                        truncation=True, return_tensors='pt')
    
    # Forward pass
    label_logits, evidence_logits = model(
        encoding['input_ids'],
        encoding['attention_mask'],
        sentence_positions
    )
    
    # Get predictions
    pred_label = ['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'][label_logits.argmax()]
    evidence_probs = torch.sigmoid(evidence_logits[0])
    
    # Apply threshold
    threshold = 0.55  # Optimized value
    pred_sentences = [i for i, prob in enumerate(evidence_probs) if prob > threshold]
    
    predictions.append({
        'id': claim.id,
        'label': pred_label,
        'evidence': {
            str(doc_id): [{
                'sentences': pred_sentences,
                'label': pred_label
            }]
        }
    })
```

### Threshold Tuning

We tested thresholds from 0.3 to 0.7:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.30 | 17.79% | 36.07% | 23.83% |
| 0.40 | 18.38% | 35.25% | 24.16% |
| **0.55** | **19.09%** | **33.06%** | **24.20%** |
| 0.70 | 23.61% | 18.58% | 20.80% |

**Optimal**: 0.55 (best F1 balance)

This is a hard problem

## Results

### Final Performance

| Metric | Score | vs. Simple Baseline |
|--------|-------|---------------------|
| **Sentence F1** | **24.20%** | **+21.64%** (9.5x improvement) |
| Precision | 19.09% | +16.43% |
| Recall | 33.06% | +30.60% |
| Abstract F1 | 80.24% | -19.76% (oracle vs. partial) |
| Label Accuracy | 0% | 0% (dev set has no labels) |

### Training Progression

| Epoch | Loss | Label Acc | Evidence Acc |
|-------|------|-----------|--------------|
| 1 | 1.276 | 63.4% | 50.0% |
| 2 | 1.198 | 65.9% | 83.3% |
| 4 | 0.910 | 81.2% | 80.0% |
| 6 | 0.662 | 93.5% | 85.7% |
| **8** | **0.507** | **97.6%** | **100%** |

**Key Observation**: Model converges well, achieving near-perfect training accuracy.

## Challenges & How I solved them

### Challenge 1: Type Mismatches (2 hours debugging)

**Problem**: Evidence dictionary keys were strings but accessed as integers.

**Solution**:
```python
# What was wrong initially
doc_id = int(list(claim.evidence.keys())[0])
if doc_id in claim.evidence:  # KeyError!

# CORRECTION made now
doc_id_str = list(claim.evidence.keys())[0]  # Keep as string
doc_id_int = int(doc_id_str)                  # Convert for corpus
if doc_id_str in claim.evidence:              # Use string for evidence
```

### Challenge 2: Misleading Metrics (3 hours)

**Problem**: Evidence accuracy showed 90%+ but model wasn't learning at all

**Root Cause**: Counted all 20 positions (including padding) as valid.

**Solution**:
```python
# What I did initially WRONG
valid_positions = (evidence_mask >= 0).float()  # All positions

# CORRECTION made now
valid_positions = (sentence_positions > 0).float()  # Only real sentences
```

After fix: Accuracy showed real progression (50% → 100%).

### Challenge 3: Low Precision (ongoing)

**Problem**: 19% precision = 81% false positives.

**Root Cause**: Class imbalance (1-2 evidence sentences per 10-sentence abstract).

**Attempted Solutions**:
- Threshold tuning: Marginal improvement
- Evidence loss weight (1.0 → 2.0): Helped with recall
- Hard negative mining: Planned for Milestone 3 redesign needed
- Focal loss: Planned for Milestone 3

## Comparison to Published Baseline

| Aspect | Our Implementation | Wadden et al. (2020) | Gap |
|--------|-------------------|----------------------|-----|
| Sentence F1 | 24.20% | 46.9% | -22% |
| Training | 8 epochs, 7 min | Extensive tuning | - |
| Retrieval | Oracle (100%) | BM25 (~65%) | +35% |
| Negatives | None | Hard negatives | Missing |
| Loss | BCE | Focal loss (likely) | Different |

**Analysis**: Gap is expected due to limited training resources and missing components (hard negatives, advanced loss functions).

## File Structure

```
src/
├── claim_verification/
│   ├── model.py              # SciBERT arch will export later
│   ├── simple_baseline.py    # Simple baseline
│   └── inference.py          # Generate predictions TODO
├── common/
│   └── data_utils.py         # Data loading utilities
└── evaluation/
    └── score_claims.py       # Evaluation script

notebooks/
└── train_scibert_baseline.ipynb  # Training notebook (Colab-ready) MAIN WORK

models/
└── claim_verifier/
    ├── model_epoch1.pt
    ├── model_epoch2.pt
    └── ....
```

## Next Steps (Milestone 3)

### Extension 1: Hard Negative Mining
Add NOT_ENOUGH_INFO examples with random documents:
```python
# Add 50% negatives to training data
neg_ratio = 0.5
for claim in nei_claims:
    random_doc = random.choice(corpus)
    # Train with all-zero evidence mask
```

**Expected**: +5-8% F1 (better precision)

### Extension 2: BM25 Retrieval
Replace oracle with realistic retrieval:
```python
from rank_bm25 import BM25Okapi

# Build index
bm25 = BM25Okapi(corpus_texts)
top_docs = bm25.get_top_n(claim, corpus, n=5)
```

**Expected**: 65-70% abstract F1, 35-40% sentence F1

### Extension 3: Focal Loss
Handle class imbalance:
```python
focal_loss = alpha * (1-pt)^gamma * BCE_loss
```

**Expected**: +3-5% F1 (better precision)

### Target Performance
With extensions: **35-40% sentence F1** (approaching published baseline)

What I found out the hard way : 

SciFact is super hard problem and needle in haystack problem
So, we Need evidence at sentence level and abstract-level won't get us to State of the Art

So even if the F-1 score is around 26% currently, we target to get it up to 35% using advanced techniques in next milestone. This is a good result considering the SciFact paper actually got just 46.9% accuracy after many years of research.


## References

1. Wadden et al. (2020). "Fact or Fiction: Verifying Scientific Claims." EMNLP.
2. Beltagy et al. (2019). "SciBERT: A Pretrained Language Model for Scientific Text." EMNLP.
3. Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.