"""
Shared data loading utilities for all components.
Handles SciFact data format with proper type conversions.
"""
import jsonlines
from typing import List, Dict, Any
from pathlib import Path


class Claim:
    """Represents a scientific claim with evidence."""
    
    def __init__(self, id: int, claim: str, evidence: Dict = None, label: str = None, cited_doc_ids: List = None):
        self.id = id
        self.claim = claim
        self.evidence = evidence or {}
        self.label = label
        self.cited_doc_ids = cited_doc_ids or []
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'claim': self.claim,
            'evidence': self.evidence,
            'label': self.label,
            'cited_doc_ids': self.cited_doc_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """
        Parse claim from JSONL format.
        Handles label inference from evidence if top-level label is missing.
        """
        raw_label = data.get('label')
        evidence = data.get('evidence', {})

        # Infer label if not provided
        if raw_label is not None:
            label = raw_label
        else:
            # Extract labels from evidence
            labels = []
            for doc_ev in evidence.values():
                for sent in doc_ev:
                    labels.append(sent.get('label'))
            
            # Priority: CONTRADICT > SUPPORT > NOT_ENOUGH_INFO
            if 'CONTRADICT' in labels:
                label = 'CONTRADICT'
            elif 'SUPPORT' in labels:
                label = 'SUPPORT'
            else:
                label = 'NOT_ENOUGH_INFO'

        return cls(
            id=data['id'],
            claim=data['claim'],
            evidence=evidence,
            label=label,
            cited_doc_ids=data.get('cited_doc_ids', [])
        )


class Document:
    """Represents a scientific paper abstract."""
    
    def __init__(self, doc_id: int, title: str, abstract: List[str]):
        # Store doc_id as integer
        self.doc_id = int(doc_id)
        self.title = title
        self.abstract = abstract  # List of sentences
    
    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'abstract': self.abstract
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            doc_id=int(data['doc_id']),  # Ensure integer
            title=data['title'],
            abstract=data['abstract']
        )


def load_claims(filepath: str) -> List[Claim]:
    """Load claims from JSONL file."""
    claims = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            claims.append(Claim.from_dict(obj))
    return claims


def load_corpus(filepath: str) -> Dict[int, Document]:
    """
    Load corpus of documents.
    Returns dict with INTEGER keys for consistent type handling.
    """
    corpus = {}
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            doc = Document.from_dict(obj)
            corpus[doc.doc_id] = doc  # Integer key
    return corpus


def save_predictions(predictions: List[Dict], filepath: str):
    """Save predictions to JSONL file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(filepath, mode='w') as writer:
        writer.write_all(predictions)


if __name__ == "__main__":
    # Test the data utilities
    print("Testing data utilities...")
    
    # Test loading
    try:
        claims = load_claims('data/scifact/data/claims_train.jsonl')
        corpus = load_corpus('data/scifact/data/corpus.jsonl')
        
        print(f" Loaded {len(claims)} claims")
        print(f" Loaded {len(corpus)} documents")
        
        # Test claim
        example = claims[0]
        print(f"\nExample claim:")
        print(f"  ID: {example.id}")
        print(f"  Claim: {example.claim[:100]}...")
        print(f"  Label: {example.label}")
        print(f"  Cited docs: {example.cited_doc_ids}")
        
        # Test document
        if example.cited_doc_ids:
            doc_id = int(example.cited_doc_ids[0])
            if doc_id in corpus:
                doc = corpus[doc_id]
                print(f"\nExample document:")
                print(f"  ID: {doc.doc_id} (type: {type(doc.doc_id)})")
                print(f"  Title: {doc.title}")
                print(f"  Sentences: {len(doc.abstract)}")
        
        print("\nAll tests passed!")
        
    except FileNotFoundError as e:
        print(f"Files not found: {e}")
        print("  Make sure you're in the project root directory")