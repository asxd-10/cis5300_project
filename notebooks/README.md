# Notebooks Organization

This directory contains training notebooks for different models and extensions.

## Structure

### Claim Verification (SciFact Task)

1. **`train_scibert_baseline.ipynb`**
   - Milestone 2 strong baseline
   - SciBERT multi-task architecture
   - Performance: 24.20% F1

2. **`train_scibert_ext1_hard_negatives.ipynb`**
   - Extension 1: Hard Negative Mining for SciBERT
   - Includes hyperparameter tuning grid search
   - Status: In progress (initial config showed -2.36% F1, tuning ongoing)

3. **`train_pub_med_bert.ipynb`**
   - PubMedBERT sentence-pair baseline
   - Original SciFact architecture
   - Performance: 39.30% F1 (best result so far)

4. **`train_pubmedbert_ext1_hard_negatives.ipynb`**
   - Extension 1: Hard Negative Mining for PubMedBERT
   - Status: Planned (not yet implemented)

5. **`train_gnn_extension.ipynb`** (Future)
   - Graph Neural Network extension
   - Status: Planned for final milestone

### Section Classification

1. **`train_scibert_section_classification.ipynb`**
   - Section classification baseline
   - Performance: 88.15% accuracy, 82.40% macro-F1

## Experiment Log

### SciBERT Experiments

| Experiment | Config | F1 | Precision | Recall | Notes |
|------------|--------|-----|-----------|--------|-------|
| Baseline | No negatives, w=2.0 | 24.20% | 19.09% | 33.06% | Milestone 2 |
| Ext1 Initial | neg=0.3, w=1.5 | 21.84% | 18.71% | 26.23% | Too conservative |
| Ext1 Grid Search | Various | TBD | TBD | TBD | In progress |

### PubMedBERT Experiments

| Experiment | Config | F1 | Precision | Recall | Notes |
|------------|--------|-----|-----------|--------|-------|
| Baseline | Sentence-pair | 39.30% | 28.21% | 64.75% | Best result |

## Running Notebooks

All notebooks are designed to run in Google Colab with GPU support.

1. Mount Google Drive
2. Clone repository
3. Install dependencies
4. Run cells sequentially

See individual notebook markdown cells for detailed instructions.

## Best Practices

- Always save model checkpoints after each epoch
- Document hyperparameters in notebook markdown
- Compare results with baseline in results summary cells
- Use threshold tuning for optimal F1

