# SciFact Dataset

## Overview
SciFact is a dataset for verifying scientific claims against a corpus of biomedical research abstracts.

## Source
- **Paper:** Wadden et al., "Fact or Fiction: Verifying Scientific Claims" (EMNLP 2020)
- **Download:** https://scifact.s3.us-west-2.amazonaws.com/release/2020-05-01/data.tar.gz
- **GitHub:** https://github.com/allenai/scifact

## Dataset Statistics

### Claims
| Split | # Claims | # SUPPORTS | # REFUTES | # NEI |
|-------|----------|------------|-----------|-------|
| Train | 809      | 269 (33%)  | 112 (14%) | 428 (53%) |
| Dev   | 300      | 102 (34%)  | 38 (13%)  | 160 (53%) |
| Test  | 300      | 115 (38%)  | 36 (12%)  | 149 (50%) |

### Corpus
- **Total abstracts:** 5,183
- **Domain:** Biomedical research (from PubMed)
- **Average abstract length:** 11.1 sentences
- **Average sentences per rationale:** 2.1

## Data Format

### Claims File (claims_train.jsonl, claims_dev.jsonl, claims_test.jsonl)

Each line is a JSON object:
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
- `claim`: The scientific claim text
- `evidence`: Dictionary mapping document IDs to evidence
  - `sentences`: 0-indexed sentence positions in the abstract
  - `label`: "SUPPORT" or "CONTRADICT"
- `cited_doc_ids`: List of relevant document IDs (for retrieval evaluation)

**Note:** Test set has `cited_doc_ids` but no `evidence` field (blind evaluation)

### Corpus File (corpus.jsonl)

Each line is a JSON object representing one abstract:
```json
{
  "doc_id": 14717500,
  "title": "Acupuncture for depression: a randomised controlled trial",
  "abstract": [
    "OBJECTIVE: To determine the effectiveness of acupuncture for depression.",
    "DESIGN: Randomised controlled trial.",
    "SETTING: University teaching hospital.",
    "PARTICIPANTS: 70 patients with major depression.",
    "...",
    "CONCLUSIONS: Acupuncture was not superior to sham acupuncture."
  ],
  "structured": true
}
```

**Fields:**
- `doc_id`: PubMed ID of the paper
- `title`: Paper title
- `abstract`: List of sentences (0-indexed)
- `structured`: Boolean indicating if abstract has section structure

## Example

**Claim:** "Acupuncture is not effective for treating depression."

**Evidence from doc 14717500:**
- Sentence 3: "PARTICIPANTS: 70 patients with major depression."
- Sentence 5: "CONCLUSIONS: Acupuncture was not superior to sham acupuncture."

**Label:** SUPPORT (the evidence supports the claim)

## Data Characteristics

### Label Distribution
- **Class imbalance:** NOT ENOUGH INFO is majority class (~51%)
- **Challenging:** Even SUPPORTS is only ~36% of data
- **Rare class:** REFUTES is only ~13% of data

### Claim Characteristics
- **Average claim length:** 17.3 words
- **Claim types:** Treatment efficacy, causal relationships, statistical findings
- **Reasoning required:** Most claims need multiple evidence sentences

### Evidence Characteristics
- **Average evidence sentences per claim:** 2.1
- **Evidence can be implicit:** Sometimes requires domain knowledge
- **Multi-document reasoning:** Some claims need evidence from multiple papers

## Citation
```bibtex
@inproceedings{wadden2020scifact,
  title={Fact or Fiction: Verifying Scientific Claims},
  author={Wadden, David and Lin, Shanchuan and Lo, Kyle and Wang, Lucy Lu and van Zuylen, Madeleine and Cohan, Arman and Hajishirzi, Hannaneh},
  booktitle={EMNLP},
  year={2020}
}
```