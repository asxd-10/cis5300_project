# Majority-Class Baseline for PubMed RCT Section Classification

## Overview

This is a **simple baseline** for the section classification task on the PubMed 200K RCT dataset.  
The baseline predicts the **majority class** from the training data for all sentences in the test set.  

- **Purpose:** Provides a minimal benchmark for evaluation.  
- **Expected behavior:** Accuracy is approximately the proportion of the majority class (~33% for METHODS).  
- **Metrics:** Accuracy and Macro-F1.

---

## Code

The baseline is implemented in `majority_baseline.py`:

```bash
cis5300_project/src/section_classification/majority_baseline.py
