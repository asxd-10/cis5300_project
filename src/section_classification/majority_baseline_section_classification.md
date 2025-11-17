# Majority-Class Baseline for PubMed RCT Section Classification

## Overview

This is a **simple baseline** for the section classification task on the PubMed 200K RCT dataset.  
The baseline predicts the **majority class** from the training data for all sentences in the test set.  

- **Purpose:** Provides a minimal benchmark for evaluation.  
- **Expected behavior:** Accuracy is approximately the proportion of the majority class (~33% for METHODS).  
- **Metrics:** Accuracy and Macro-F1.

---

## Code Location

The baseline is implemented in:

cis5300_project/src/section_classification/majority_baseline.py


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
python src/section_classification/majority_baseline.py \
    data/pubmed_rct/train.txt \
    data/pubmed_rct/test.txt

train.txt → training data

test.txt → test data
The script will automatically create:
baseline_predictions.txt

This file contains one predicted label per line for the test set.

Example Output
Majority label: METHODS
Accuracy: 0.3298
Macro-F1: 0.0992
Predictions saved to baseline_predictions.txt
