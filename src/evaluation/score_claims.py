"""
Evaluation script for claim verification.
Computes abstract-level F1 and sentence-level F1.
"""

import argparse
import jsonlines
from typing import Dict, Tuple


def load_predictions(filepath: str) -> Dict:
    """Load predictions from JSONL file."""
    predictions = {}
    with jsonlines.open(filepath) as reader:
        for pred in reader:
            predictions[pred['id']] = pred
    return predictions


def load_gold(filepath: str) -> Dict:
    """Load gold annotations from JSONL file."""
    gold = {}
    with jsonlines.open(filepath) as reader:
        for item in reader:
            gold[item['id']] = item
    return gold


def normalize_doc_id(doc_id) -> int:
    """Convert doc_id to integer for consistent comparison."""
    return int(doc_id)


def compute_abstract_f1(gold: Dict, predictions: Dict) -> Tuple[float, float, float]:
    """
    Compute abstract-level F1: did we retrieve the right papers?
    """
    tp = fp = fn = 0
    
    for claim_id, gold_item in gold.items():
        if claim_id not in predictions:
            # Prediction missing for this claim
            if 'cited_doc_ids' in gold_item and gold_item['cited_doc_ids']:
                fn += len(gold_item['cited_doc_ids'])
            continue
        
        pred_item = predictions[claim_id]
        
        # Gold cited docs (normalize to int)
        gold_docs = set()
        for doc_id in gold_item.get('cited_doc_ids', []):
            gold_docs.add(normalize_doc_id(doc_id))
        
        # Predicted docs (normalize to int)
        pred_docs = set()
        if 'evidence' in pred_item and pred_item['evidence']:
            for doc_id in pred_item['evidence'].keys():
                pred_docs.add(normalize_doc_id(doc_id))
        
        # Compute TP, FP, FN
        tp += len(gold_docs & pred_docs)
        fp += len(pred_docs - gold_docs)
        fn += len(gold_docs - pred_docs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def compute_sentence_f1(gold: Dict, predictions: Dict) -> Tuple[float, float, float]:
    """
    Compute sentence-level F1: did we find the right evidence with correct label?
    This is the PRIMARY metric for SciFact.
    """
    tp = fp = fn = 0
    
    for claim_id, gold_item in gold.items():
        if 'evidence' not in gold_item or not gold_item['evidence']:
            continue
        
        if claim_id not in predictions:
            # Count all gold evidence as FN
            for doc_id, evidence_list in gold_item['evidence'].items():
                for ev in evidence_list:
                    fn += len(ev['sentences'])
            continue
        
        pred_item = predictions[claim_id]
        
        # For each document in gold evidence
        for doc_id, gold_evidence_list in gold_item['evidence'].items():
            doc_id_int = normalize_doc_id(doc_id)
            
            for gold_ev in gold_evidence_list:
                gold_sentences = set(gold_ev['sentences'])
                gold_label = gold_ev['label']
                
                # Check if this doc is in predictions (handle both int and str keys)
                pred_evidence_list = None
                if 'evidence' in pred_item:
                    # Try integer key first, then string key
                    if doc_id_int in pred_item['evidence']:
                        pred_evidence_list = pred_item['evidence'][doc_id_int]
                    elif str(doc_id) in pred_item['evidence']:
                        pred_evidence_list = pred_item['evidence'][str(doc_id)]
                
                if pred_evidence_list is None:
                    fn += len(gold_sentences)
                    continue
                
                # Find matching evidence entry (same label)
                matched = False
                for pred_ev in pred_evidence_list:
                    pred_sentences = set(pred_ev['sentences'])
                    pred_label = pred_ev['label']
                    
                    if pred_label == gold_label:
                        # Count sentence overlap
                        tp += len(gold_sentences & pred_sentences)
                        fp += len(pred_sentences - gold_sentences)
                        fn += len(gold_sentences - pred_sentences)
                        matched = True
                        break
                
                if not matched:
                    # No matching label found
                    fn += len(gold_sentences)
                    # Count all predicted sentences as FP
                    for pred_ev in pred_evidence_list:
                        fp += len(pred_ev['sentences'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def compute_label_accuracy(gold: Dict, predictions: Dict) -> float:
    """
    Compute label-only accuracy (ignoring evidence).
    """
    correct = 0
    total = 0
    
    for claim_id, gold_item in gold.items():
        if 'label' not in gold_item:
            continue
        
        if claim_id not in predictions or 'label' not in predictions[claim_id]:
            total += 1
            continue
        
        gold_label = gold_item['label']
        pred_label = predictions[claim_id]['label']
        
        if gold_label == pred_label:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Evaluate claim verification predictions')
    parser.add_argument('--gold', required=True, help='Gold annotations file (.jsonl)')
    parser.add_argument('--predictions', required=True, help='Predictions file (.jsonl)')
    args = parser.parse_args()
    
    print("="*60)
    print("CLAIM VERIFICATION EVALUATION")
    print("="*60)
    
    print("\nLoading data...")
    gold = load_gold(args.gold)
    predictions = load_predictions(args.predictions)
    
    print(f"  Gold claims: {len(gold)}")
    print(f"  Predictions: {len(predictions)}")
    
    # Compute metrics
    print("\nComputing metrics...")
    abs_p, abs_r, abs_f1 = compute_abstract_f1(gold, predictions)
    sent_p, sent_r, sent_f1 = compute_sentence_f1(gold, predictions)
    label_acc = compute_label_accuracy(gold, predictions)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nAbstract-level (Retrieval):")
    print(f"  Precision: {abs_p:.4f}")
    print(f"  Recall:    {abs_r:.4f}")
    print(f"  F1:        {abs_f1:.4f}")
    
    print(f"\nSentence-level (Evidence + Label): PRIMARY METRIC")
    print(f"  Precision: {sent_p:.4f}")
    print(f"  Recall:    {sent_r:.4f}")
    print(f"  F1:        {sent_f1:.4f}")
    
    print(f"\nLabel-only:")
    print(f"  Accuracy:  {label_acc:.4f}")
    
    print("="*60)
    
    # Interpretation
    print("\nInterpretation:")
    if abs_f1 > 0.9:
        print(" Retrieval is excellent (oracle or near-oracle)")
    elif abs_f1 > 0.5:
        print("  Retrieval is working reasonably")
    else:
        print("  Retrieval needs improvement")
    
    if sent_f1 < 0.1:
        print(" Evidence extraction is very weak (baseline level)")
    elif sent_f1 < 0.3:
        print("  Evidence extraction is improving but below target")
    elif sent_f1 < 0.45:
        print("  Evidence extraction is competitive")
    else:
        print("  Evidence extraction is state-of-the-art!")
    
    print("="*60)
    
    # Return dict for programmatic use
    return {
        'abstract_f1': abs_f1,
        'sentence_f1': sent_f1,
        'label_accuracy': label_acc
    }


if __name__ == "__main__":
    main()