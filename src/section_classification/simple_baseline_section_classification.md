# Simple Baseline for PubMed RCT Section Classification

## Overview

This is a **simple baseline** for the section classification task on the PubMed 200K RCT dataset.  
The baseline predicts the **majority class** from the training data for all sentences in the **dev and test sets**.

- **Purpose:** Provides a minimal benchmark for evaluation.  
- **Expected behavior:** Accuracy is approximately the proportion of the majority class (~33% for METHODS).  
- **Metrics:** Accuracy and Macro-F1.

---

## Code Location

The baseline is implemented in:

cis5300_project/src/section_classification/simple_baseline_section_classification.py


Key features:

1. Loads training and test data from tab-separated files (`LABEL<TAB>sentence`).  
2. Finds the **majority label** in the training data.  
3. Predicts the **majority label** for all test sentences.  
4. Computes **accuracy** and **Macro-F1**.  
5. Saves predictions to a file: `baseline_predictions.txt`.

---

## How to Run
From the **project root**, execute:

```bash
python src/section_classification/simple_baseline_section_classification.py \
    data/pubmed_rct/train.txt \
    data/pubmed_rct/dev.txt \
    data/pubmed_rct/test.txt
```

train.txt → training data

test.txt → test data
The script will automatically create:

simple_baseline_dev_predictions.txt

simple_baseline_test_predictions.txt

Each file contains one predicted label per line.


```
Example Output:
Majority label: METHODS
Dev Accuracy: 0.3298
Dev Macro-F1: 0.0992
Predictions saved to simple_baseline_dev_predictions.txt & simple_baseline_test_predictions.txt
```
