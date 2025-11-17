#!/usr/bin/env python3
"""
Simple Baseline for PubMed 200K RCT Section Classification
TF-IDF + Majority-Class Predictor
"""

import sys
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------
# Label mapping and preprocessing
# ------------------------------
label2id = {
    "BACKGROUND": 0,
    "OBJECTIVE": 1,
    "METHODS": 2,
    "RESULTS": 3,
    "CONCLUSIONS": 4
}
id2label = {v: k for k, v in label2id.items()}

def preprocess(text):
    """Basic preprocessing: strip and lowercase."""
    return text.strip().lower()


def load_pubmed_rct(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("###"):
                continue
            label, sentence = line.split("\t", 1)
            data.append((label, sentence))
    return data


def majority_baseline_predict(length, majority_label):
    return [majority_label] * length


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python simple_baseline_section_classification.py train.txt dev.txt test.txt")
        sys.exit(1)

    train_path, dev_path, test_path = sys.argv[1:]

    train_data = load_pubmed_rct(train_path)
    dev_data   = load_pubmed_rct(dev_path)
    test_data  = load_pubmed_rct(test_path)

    train_texts = [preprocess(s) for _, s in train_data]
    train_labels = [label for label, _ in train_data]

    dev_labels = [label for label, _ in dev_data]
    test_labels = [label for label, _ in test_data]

    # TF-IDF (not used for prediction, just part of baseline pipeline)
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=50000)
    vectorizer.fit(train_texts)

    majority_label = Counter(train_labels).most_common(1)[0][0]
    print("Majority label (train set):", majority_label)

    pred_dev  = majority_baseline_predict(len(dev_labels), majority_label)
    pred_test = majority_baseline_predict(len(test_labels), majority_label)


    acc_dev = accuracy_score(dev_labels, pred_dev)
    macro_f1_dev = f1_score(dev_labels, pred_dev, average="macro")

    print("Dev Accuracy:", acc_dev)
    print("Dev Macro-F1:", macro_f1_dev)


    with open("simple_baseline_dev_predictions.txt", "w") as f:
        for p in pred_dev:
            f.write(f"{p}\n")

    with open("simple_baseline_test_predictions.txt", "w") as f:
        for p in pred_test:
            f.write(f"{p}\n")

    print("Predictions saved: simple_baseline_dev_predictions.txt & simple_baseline_test_predictions.txt")
