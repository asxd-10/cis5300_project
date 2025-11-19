# Simple Baseline for Claim Verification

## Overview

The simple baseline establishes floor performance using three naive heuristics:
1. **Oracle Retrieval**: Use gold `cited_doc_ids` (perfect document retrieval)
2. **Majority Class Prediction**: Predict "SUPPORT" for all claims
3. **First Sentence Heuristic**: Guess sentence index 0 as evidence

**Purpose**: Shows that even with perfect retrieval, evidence extraction is challenging.

## Requirements

```bash
pip install jsonlines scikit-learn
```

## Usage

### Basic Usage

```bash
python src/claim_verification/simple_baseline.py --split dev
```

### Options

```
--data_dir    Data directory (default: data/scifact/data)
--output      Output file path (default: output/dev/simple_baseline.jsonl)
--split       Data split to process (choices: train, dev, test)
```

### Examples

**Process dev set:**
```bash
python src/claim_verification/simple_baseline.py \
  --split dev \
  --output output/dev/simple_baseline.jsonl
```

**Process test set:**
```bash
python src/claim_verification/simple_baseline.py \
  --split test \
  --output output/test/simple_baseline.jsonl
```

## Example Output

```
============================================================
SIMPLE BASELINE: Oracle Retrieval + Majority Class
============================================================

Loading data from data/scifact/data...
  300 dev claims
  5183 abstracts in corpus
  809 training claims

Train label distribution:
  SUPPORT             :  332 ( 41.0%)
  NOT_ENOUGH_INFO     :  304 ( 37.6%)
  CONTRADICT          :  173 ( 21.4%)

Majority label: SUPPORT
Strategy: Use gold docs + guess first sentence + predict SUPPORT

Generating predictions for 300 claims...
  Progress: 300/300

Saved predictions to output/dev/simple_baseline.jsonl
Done!

============================================================
NEXT STEP: Run evaluation
============================================================
python src/evaluation/score_claims.py \
  --gold data/scifact/data/claims_dev.jsonl \
  --predictions output/dev/simple_baseline.jsonl
============================================================
```

## Algorithm

```python
def predict_claim_oracle(claim, corpus, majority_label):
    """
    Oracle baseline prediction
    
    Args:
        claim: Claim object with cited_doc_ids
        corpus: Document corpus
        majority_label: Most common label in training (SUPPORT)
    
    Returns:
        Prediction dict in SciFact format
    """
    evidence = {}
    
    # Use oracle retrieval (gold documents)
    if claim.cited_doc_ids:
        for doc_id in claim.cited_doc_ids:
            doc_id_int = int(doc_id)
            
            if doc_id_int in corpus and corpus[doc_id_int].abstract:
                # Heuristic: guess first sentence
                evidence[doc_id_int] = [{
                    'sentences': [0],
                    'label': majority_label
                }]
    
    return {
        'id': claim.id,
        'label': majority_label,
        'evidence': evidence
    }
```

## Results

| Metric | Score | Analysis |
|--------|-------|----------|
| **Sentence F1** | **2.56%** | Very low - first sentence rarely contains evidence |
| **Abstract F1** | **100%** | Perfect - using gold documents (oracle) |
| **Label Accuracy** | **0%** | All predictions are SUPPORT, but dev set has hidden labels |
| **Precision** | 2.66% | 97% of predicted sentences are wrong |
| **Recall** | 2.46% | Missing 97.5% of actual evidence |

## Key Insights

1. **Perfect retrieval ≠ good performance**: Even with 100% abstract F1, sentence F1 is only 2.56%
2. **Evidence extraction is the bottleneck**: Finding the right sentences is much harder than finding the right papers
3. **First sentence heuristic fails**: Evidence is typically in RESULTS/CONCLUSIONS, not the first sentence
4. **Single label doesn't work**: Real claims have mixed evidence (some SUPPORT, some CONTRADICT)

## Comparison to Random

| Method | Sentence F1 |
|--------|-------------|
| Random guessing | ~0.5-1% |
| **Simple baseline** | **2.56%** |
| SciBERT (our impl) | 24.20% |
| Published baseline | 46.9% |

**Takeaway**: Simple baseline is ~2.5x better than random but 18x worse than a trained model.

## Next Steps

After running simple baseline:

1. **Evaluate** to get baseline numbers:
   ```bash
   python src/evaluation/score_claims.py \
     --gold data/scifact/data/claims_dev.jsonl \
     --predictions output/dev/simple_baseline.jsonl
   ```

2. **Train strong baseline** (see `strong-baseline.md`):
   ```bash
   # Should achieve 20-25% F1 (9-10x improvement)
   ```

## File Structure

```
src/claim_verification/
├── simple_baseline.py       # This script
└── model.py                 # SciBERT model (for strong baseline)

output/dev/
└── simple_baseline.jsonl    # Generated predictions
```

## Code Structure

The simple baseline consists of:

1. **Data Loading** (`load_claims`, `load_corpus`)
2. **Majority Label Calculation** (from training set)
3. **Oracle Retrieval** (use `cited_doc_ids`)
4. **Heuristic Prediction** (first sentence + majority label)
5. **Output Generation** (JSONL format)
