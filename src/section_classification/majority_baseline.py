#!/usr/bin/env python3
import sys
from collections import Counter

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

def majority_baseline_predict(data, majority_label):
    return [majority_label] * len(data)

def accuracy(gold, pred):
    correct = sum(g == p for g, p in zip(gold, pred))
    return correct / len(gold)

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = load_pubmed_rct(train_path)
    test_data  = load_pubmed_rct(test_path)

    # Find majority label in training data
    label_counts = Counter([label for label, sent in train_data])
    majority_label = label_counts.most_common(1)[0][0]

    # Predict
    gold = [label for label, sent in test_data]
    pred = majority_baseline_predict(test_data, majority_label)

    # Score
    print("Majority label:", majority_label)
    print("Accuracy:", accuracy(gold, pred))
