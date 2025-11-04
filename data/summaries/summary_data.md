# Evidence Summarization Dataset 
TODO (Ashay) : Explore alternative approaches as manual annotation does not seem feasible
Either way, this is only to be done once claim verification and section tasks are done (Not priority)

## Overview
Natural language summaries of claim verification decisions, derived from SciFact.

## Source
- **Base data:** SciFact claims and evidence
- **Generation:** We create summaries from claim + evidence + verdict
- **Purpose:** Training BART to generate human-readable explanations

## Dataset Creation

We will create summaries in two ways:

### 1: Template-Based
```python
# Example template
if label == "SUPPORTS":
    summary = f"The claim is supported by {num_papers} paper(s). "
    summary += f"Key evidence: {evidence_snippet}"
elif label == "REFUTES":
    summary = f"The claim is refuted by {num_papers} paper(s). "
    summary += f"Contradicting evidence: {evidence_snippet}"
```

### 2: Manual Annotation (50-100 examples)
Team members will write gold summaries for a subset of claims.

## Target Format
```json
{
  "claim_id": 2,
  "claim": "Acupuncture is not effective for treating depression.",
  "verdict": "SUPPORTS",
  "evidence_docs": [14717500],
  "evidence_sentences": [
    "CONCLUSIONS: Acupuncture was not superior to sham acupuncture."
  ],
  "summary": "The claim is supported by a randomized controlled trial of 70 patients with major depression, which found that acupuncture was not superior to sham acupuncture for treating depression."
}
```

## Statistics (Target)

| Split | # Summaries |
|-------|-------------|
| Train | 600-700     |
| Dev   | 200-250     |
| Test  | 200-250     |

## Evaluation Metrics

- **ROUGE-L:** Overlap with reference summaries
- **BERTScore:** Semantic similarity
- **Human eval:** Faithfulness (does summary match evidence?) and Informativeness (is it useful?)