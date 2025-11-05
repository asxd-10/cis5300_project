# PubMed 200K RCT Dataset

## Overview
Sequential sentence classification for medical abstracts with section labels.

## Source
- **Paper:** Dernoncourt & Lee, "PubMed 200k RCT" (IJCNLP 2017)
- **GitHub:** https://github.com/Franck-Dernoncourt/pubmed-rct

## Verified Statistics

| Split | # Abstracts | # Sentences | #Lines |
|-------|-------------|-------------|--------|
| Train | 15,000      | 180,040     | 210,040|
| Dev   | 2,500       | 30,212      | 35,212 |
| Test  | 2,500       | 30,135      | 35,135 |
| **Total** | **20,000** | **240,387** | **280,387** |

These line counts are estimates using bash functions - as Line counts include abstract markers.

### Label Distribution (Train Split)

| Label       | Count    | Percentage |
|-------------|----------|------------|
| METHODS     | 59,353   | 33.0%      |
| RESULTS     | 57,953   | 32.2%      |
| CONCLUSIONS | 27,168   | 15.1%      |
| BACKGROUND  | 21,727   | 12.1%      |
| OBJECTIVE   | 13,839   | 7.7%       |

**Key observations:**
- METHODS and RESULTS dominate (~65% combined)
- RESULTS and CONCLUSIONS most relevant for evidence
- Consistent distribution across splits

## Data Format

Plain text, tab-separated:
```
###24293578
BACKGROUND	Emotional eating is associated with overeating and obesity.
OBJECTIVE	This study tests if attention bias moderates emotional eating...
METHODS	Participants were 85 undergraduate students...
RESULTS	Food attention bias moderated the effect...
CONCLUSIONS	Attentional bias for food relates to emotional eating.
```

**Format:**
- `###` + PubMed ID starts each abstract
- Each line: `LABEL<tab>sentence`

## Why This Dataset

**Purpose:** Train section classifier (Component 2) to identify evidence-rich sections in retrieved papers.

**Key insight:** RESULTS and CONCLUSIONS sections contain most evidence for claim verification. METHODS and BACKGROUND rarely contain supporting/refuting evidence.

**Use in project:** Our trained section classifier filters SciFact papers to focus claim verification on relevant sections, reducing noise and improving accuracy.

## Citation
```bibtex
@inproceedings{dernoncourt2017pubmed,
  title={PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts},
  author={Dernoncourt, Franck and Lee, Ji Young},
  booktitle={IJCNLP},
  year={2017}
}
```