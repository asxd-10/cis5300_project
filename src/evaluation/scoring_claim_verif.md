# SciFact Claim Verification - Evaluation Metrics

## Overview
Describes the evaluation metrics for the scientific claim verification task and how to use the evaluation script.

## Metrics

### Primary Metric: Sentence-Level F1

**Definition**: Measures whether we correctly identify evidence sentences with the correct label (SUPPORT/CONTRADICT).

**Calculation**:
- **True Positive (TP)**: Predicted sentence appears in gold evidence with matching label
- **False Positive (FP)**: Predicted sentence not in gold evidence, or wrong label
- **False Negative (FN)**: Gold evidence sentence not predicted

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Why it's the primary metric**: It evaluates the complete task - retrieving the right documents, extracting the right sentences, AND predicting the correct verdict.

### Secondary Metrics

**Abstract-Level F1**: Measures document retrieval quality
- Did we retrieve the right scientific papers?
- 100% with oracle retrieval, lower with BM25

**Label Accuracy**: Classification accuracy (ignoring evidence)
- Note: Dev/test sets have no top-level labels, so this will be 0% for evaluation

## Usage

### Basic Usage

```bash
python src/evaluation/score_claims.py \
  --gold data/scifact/data/claims_dev.jsonl \
  --predictions output/dev/predictions.jsonl
```

### Example with Simple Baseline

```bash
# Run simple baseline
python src/claim_verification/simple_baseline.py --split dev

# Evaluate
python src/evaluation/score_claims.py \
  --gold data/scifact/data/claims_dev.jsonl \
  --predictions output/dev/simple_baseline.jsonl
```

### Example with SciBERT Baseline

```bash
# Generate predictions from trained model
# (see strong-baseline.md)

# Evaluate
python src/evaluation/score_claims.py \
  --gold data/scifact/data/claims_dev.jsonl \
  --predictions output/dev/scibert_predictions.jsonl
```

## Example Output

```
============================================================
CLAIM VERIFICATION EVALUATION
============================================================

Loading data...
  Gold claims: 300
  Predictions: 300

Computing metrics...

============================================================
RESULTS
============================================================

Abstract-level (Retrieval):
  Precision: 1.0000
  Recall:    1.0000
  F1:        1.0000

Sentence-level (Evidence + Label): PRIMARY METRIC
  Precision: 0.2420
  Recall:    0.3306
  F1:        0.2420

Label-only:
  Accuracy:  0.0000
============================================================

Interpretation:
  ✓ Retrieval is excellent (oracle or near-oracle)
  → Evidence extraction is improving but below target
============================================================
```

## Prediction Format

Your predictions file must be in JSONL format with the following structure:

```json
{
  "id": 1,
  "label": "SUPPORT",
  "evidence": {
    "14717500": [
      {
        "sentences": [2, 5],
        "label": "SUPPORT"
      }
    ]
  }
}
```

**Fields**:
- `id`: Claim ID (integer)
- `label`: Predicted verdict (SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO)
- `evidence`: Dictionary mapping document IDs (as strings) to evidence lists
  - Each evidence entry has:
    - `sentences`: List of sentence indices (0-indexed)
    - `label`: Label for this evidence (should match top-level label)

## Interpreting Results

### Good Results
- **Sentence F1 > 40%**: Competitive with published baselines
- **Sentence F1 30-40%**: Solid performance, room for improvement
- **Sentence F1 20-30%**: Basic model working, needs optimization
- **Sentence F1 < 20%**: Weak performance, check model/data

### Common Issues
- **Low Precision**: Model predicts too many sentences so Try higher threshold or hard negative mining
- **Low Recall**: Model too conservative so Try lower threshold or more training
- **High Abstract F1, Low Sentence F1**: Retrieval works but evidence extraction fails so Focus on sentence-level model
- **Label Accuracy 0%**: Dev/test sets have hidden labels - expected

## Performance Benchmarks

| Method | Sentence F1 | Abstract F1 | Notes |
|--------|-------------|-------------|-------|
| Random Guess | ~1% | ~5% | Lower bound |
| Simple Baseline (Oracle + Majority) | 2.56% | 100% | Floor with perfect retrieval |
| SciBERT (Oracle, 8 epochs) | 24.20% | 100% | Current implementation State|
| Published Baseline (Wadden et al.) | 46.9% | ~65% | With BM25 retrieval |
| Human Performance | ~80-90% | ~90% | Estimated |
