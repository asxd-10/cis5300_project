"""
Shared data loading utilities for all components
Modify as we plan data structure - this is just template
TODO (Ashay): Once individual components are done, refactor to move common code here.
"""
import json
import jsonlines
from typing import List, Dict, Any
from pathlib import Path


class Claim:
    """Represents a scientific claim with evidence."""
    
    def __init__(self, id: int, claim: str, evidence: Dict = None, label: str = None):
        self.id = id
        self.claim = claim
        self.evidence = evidence or {}
        self.label = label
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'claim': self.claim,
            'evidence': self.evidence,
            'label': self.label
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            id=data['id'],
            claim=data['claim'],
            evidence=data.get('evidence', {}),
            label=data.get('label')
        )


class Document:
    """Represents a scientific paper abstract."""
    
    def __init__(self, doc_id: str, title: str, abstract: List[str]):
        self.doc_id = doc_id
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
            doc_id=data['doc_id'],
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


def load_corpus(filepath: str) -> Dict[str, Document]:
    """Load corpus of documents."""
    corpus = {}
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            doc = Document.from_dict(obj)
            corpus[doc.doc_id] = doc
    return corpus


def save_predictions(predictions: List[Dict], filepath: str):
    """Save predictions to JSONL file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(filepath, mode='w') as writer:
        writer.write_all(predictions)


if __name__ == "__main__":
    # Test the data utilities
    print("Testing data utilities...")
    
    # Example claim
    claim = Claim(
        id=1,
        claim="Vitamin D supplementation reduces respiratory infections",
        label="SUPPORTS"
    )
    print(f"Claim: {claim.claim}")
    print(f"Label: {claim.label}")