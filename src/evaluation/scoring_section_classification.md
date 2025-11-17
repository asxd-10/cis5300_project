# Evaluation Metrics for PubMed RCT Section Classification

This document describes the **evaluation metrics** used for the section classification task and shows how to run the evaluation script `score_section_classification.py`.

---

## Metrics

We use two standard metrics:

1. **Accuracy**

Accuracy measures the fraction of correctly predicted labels:

\[
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
\]

- **Range:** 0 to 1  
- **Interpretation:** Higher accuracy indicates better overall prediction performance.

2. **Macro-F1**

Macro-F1 calculates the **F1 score for each class independently** and then averages them. It accounts for **class imbalance** and gives equal weight to all classes:

\[
\text{Macro-F1} = \frac{1}{C} \sum_{i=1}^{C} F1_i
\]

Where \(C\) is the number of classes and \(F1_i\) is the F1 score for class \(i\):

\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

- **Range:** 0 to 1  
- **Interpretation:** Higher Macro-F1 indicates better performance across all classes, even if classes are imbalanced.

---

## Evaluation Script

The evaluation script `score_section_classification.py` computes **accuracy** and **Macro-F1** given:

1. A **gold standard file** (test or dev data)  
2. A **predictions file** (produced by any model)

```bash
cis5300_project/src/evaluation/score_section_classification.py
```

File Format

Gold file: Tab-separated (LABEL<TAB>sentence) or one label per line

Predictions file: One predicted label per line

How to Run

From the project root, run:

```bash
python src/evaluation/score_section_classification.py \
    data/pubmed_rct/test.txt \
    data/pubmed_rct/baseline_predictions.txt
```
Arguments:

test.txt → gold labels

baseline_predictions.txt → model predictions

```
Example Output
Number of examples: 30135
Accuracy: 0.3298
Macro-F1: 0.0992
```

Number of examples → total number of sentences evaluated

Accuracy → fraction of correct predictions

Macro-F1 → average F1 score across all section labels
