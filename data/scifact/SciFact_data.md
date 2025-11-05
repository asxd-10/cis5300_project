# SciFact Dataset

## Overview
Dataset for verifying scientific claims against biomedical research abstracts.

## Source
- **Paper:** Wadden et al., "Fact or Fiction: Verifying Scientific Claims" (EMNLP 2020)
- **GitHub:** https://github.com/allenai/scifact

## Verified Statistics

### Claims
| Split | # Claims |
|-------|----------|
| Train | 809      |
| Dev   | 300      |
| Test  | 300      |
| **Total** | **1,409** |

### Corpus
- **Abstracts:** 5,183 PubMed abstracts
- **Avg abstract length:** 8.9 sentences
- **Avg evidence per claim:** 1.1 sentences

**Note:** Label distributions (SUPPORTS/REFUTES/NEI) will be computed during M2 baseline evaluation. Paper reports ~33% SUPPORTS, ~14% REFUTES, ~53% NEI.

## Data Format

### Claims (claims_train.jsonl, claims_dev.jsonl, claims_test.jsonl)
```json
{
  "id": 2,
  "claim": "Acupuncture is not effective for treating depression.",
  "evidence": {
    "14717500": [
      {
        "sentences": [3, 5],
        "label": "SUPPORT"
      }
    ]
  },
  "cited_doc_ids": [14717500]
}
```

**Fields:**
- `id`: Unique claim identifier
- `claim`: Scientific claim text
- `evidence`: Dict mapping doc_id → evidence list
  - `sentences`: 0-indexed positions in abstract
  - `label`: "SUPPORT" or "CONTRADICT" (maps to SUPPORTS/REFUTES)
- `cited_doc_ids`: Relevant docs for retrieval evaluation

**Note:** Test set withholds `evidence` field for blind evaluation.

### Corpus (corpus.jsonl)
```json
{
  "doc_id": 14717500,
  "title": "Acupuncture for depression: a randomised controlled trial",
  "abstract": [
    "OBJECTIVE: To determine the effectiveness of acupuncture for depression.",
    "DESIGN: Randomised controlled trial.",
    "...",
    "CONCLUSIONS: Acupuncture was not superior to sham acupuncture."
  ],
  "structured": true
}
```

**Fields:**
- `doc_id`: PubMed ID
- `title`: Paper title
- `abstract`: Sentences (0-indexed)
- `structured`: Has section headers

## Why This Dataset

**Purpose:** Primary evaluation for claim verification (Component 1).

**Key characteristics:**
- Expert-annotated by biomedical researchers
- Requires multi-sentence reasoning
- Class imbalanced (NEI is majority)
- Challenging: state-of-art is ~47% F1

**Use in project:** Train and evaluate our claim verification model. Evidence sentences guide our summarization component.

## Citation
```bibtex
@inproceedings{wadden2020scifact,
  title={Fact or Fiction: Verifying Scientific Claims},
  author={Wadden, David and Lin, Shanchuan and Lo, Kyle and Wang, Lucy Lu and van Zuylen, Madeleine and Cohan, Arman and Hajishirzi, Hannaneh},
  booktitle={EMNLP},
  year={2020}
}
```
