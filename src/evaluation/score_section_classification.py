#!/usr/bin/env python3
import sys
from sklearn.metrics import accuracy_score, f1_score

def load_labels(path):
    """
    Loads a label file.
    Expects either:
    - Tab-separated file with LABEL<TAB>text (gold)
    - Or one label per line (predictions)
    """
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # If tab exists, assume first column is label
            if "\t" in line:
                label = line.split("\t")[0]
            else:
                label = line
            labels.append(label)
    return labels

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python score.py <gold_file> <pred_file>")
        sys.exit(1)

    gold_path = sys.argv[1]
    pred_path = sys.argv[2]

    gold_labels = load_labels(gold_path)
    pred_labels = load_labels(pred_path)

    if len(gold_labels) != len(pred_labels):
        print("Error: Number of gold and predicted labels do not match!")
        sys.exit(1)

    acc = accuracy_score(gold_labels, pred_labels)
    macro_f1 = f1_score(gold_labels, pred_labels, average="macro")

    print(f"Number of examples: {len(gold_labels)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
