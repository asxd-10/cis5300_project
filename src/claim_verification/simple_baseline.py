"""
Simple baseline for claim verification.
Uses ORACLE retrieval (gold documents) + majority class prediction.
This isolates the difficulty of the verification task from retrieval.
"""
import sys
sys.path.append('.')

import argparse
import jsonlines
from collections import Counter
from src.common.data_utils import load_claims, load_corpus
from pathlib import Path


def predict_claim_oracle(claim, corpus, majority_label: str) -> dict:
    """
    Oracle baseline: use gold documents, predict majority label,
    guess first sentence as evidence.
    
    This gives us 100% retrieval accuracy to isolate verification difficulty.
    """
    evidence = {}
    
    # Use gold cited_doc_ids (oracle retrieval)
    if claim.cited_doc_ids:
        for doc_id in claim.cited_doc_ids:
            doc_id_int = int(doc_id)
            
            if doc_id_int in corpus and corpus[doc_id_int].abstract:
                # Simple heuristic: guess first sentence is evidence
                # Use majority label from training data
                evidence[doc_id_int] = [{
                    'sentences': [0],
                    'label': majority_label
                }]
    
    return {
        'id': claim.id,
        'label': majority_label,
        'evidence': evidence
    }


def main():
    parser = argparse.ArgumentParser(description='Simple oracle baseline for claim verification')
    parser.add_argument('--data_dir', default='data/scifact/data', help='Data directory')
    parser.add_argument('--output', default='output/dev/simple_baseline.jsonl', help='Output file')
    parser.add_argument('--split', default='dev', choices=['train', 'dev', 'test'], help='Data split')
    args = parser.parse_args()

    print("="*60)
    print("SIMPLE BASELINE: Oracle Retrieval + Majority Class")
    print("="*60)
    
    # Construct file paths
    claims_path = f'{args.data_dir}/claims_{args.split}.jsonl'
    train_path = f'{args.data_dir}/claims_train.jsonl'
    corpus_path = f'{args.data_dir}/corpus.jsonl'

    # Check files exist
    if not Path(train_path).exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not Path(corpus_path).exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not Path(claims_path).exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    claims = load_claims(claims_path)
    corpus = load_corpus(corpus_path)
    train_claims = load_claims(train_path)
    
    print(f" {len(claims)} {args.split} claims")
    print(f" {len(corpus)} abstracts in corpus")
    print(f" {len(train_claims)} training claims")

    # Compute majority label from training data
    train_labels = [c.label for c in train_claims if c.label]
    label_counts = Counter(train_labels)
    
    print(f"\nTrain label distribution:")
    for label, count in label_counts.most_common():
        pct = (count / len(train_labels)) * 100
        print(f"  {label:20s}: {count:4d} ({pct:5.1f}%)")
    
    majority_label = label_counts.most_common(1)[0][0]
    print(f"\n→ Majority label: {majority_label}")
    print(f"→ Strategy: Use gold docs + guess first sentence + predict {majority_label}")

    # Generate predictions
    print(f"\nGenerating predictions for {len(claims)} claims...")
    predictions = []
    
    for i, claim in enumerate(claims):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(claims)}")
        
        pred = predict_claim_oracle(claim, corpus, majority_label)
        predictions.append(pred)

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving predictions to {output_path}")
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(predictions)

    print("Done")


if __name__ == "__main__":
    main()