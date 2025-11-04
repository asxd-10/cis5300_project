# PubMed 200k RCT Dataset

## Overview
Sequential sentence classification dataset for medical abstracts with section labels.

## Source
- **Paper:** Dernoncourt & Lee, "PubMed 200k RCT" (IJCNLP 2017)
- **Download:** https://github.com/Franck-Dernoncourt/pubmed-rct
- **Files:** train.txt, dev.txt, test.txt

## Dataset Statistics

| Split | # Sentences | # Abstracts |
|-------|-------------|-------------|
| Train | 180,040     | ~15,000     |
| Dev   | 30,212      | ~2,500      |
| Test  | 30,135      | ~2,500      |

## Label Distribution

| Label       | Count    | Percentage |
|-------------|----------|------------|
| BACKGROUND  | 52,113   | 21.7%      |
| OBJECTIVE   | 31,228   | 13.0%      |
| METHODS     | 85,971   | 35.8%      |
| RESULTS     | 51,326   | 21.4%      |
| CONCLUSIONS | 19,535   | 8.1%       |

## Data Format

Plain text file with format:
```
###24293578
BACKGROUND	Emotional eating is associated with overeating and the development of obesity .
OBJECTIVE	The aim of this study was to test if attention bias for food moderates the effect of self-reported emotional eating ... 
METHODS	Participants were 85 undergraduate students ...
RESULTS	Food-related attention bias moderated the effect ...
CONCLUSIONS	Attentional bias for food relates to ...

###24854809
BACKGROUND	...
```

**Format:**
- `###` followed by PubMed ID starts each abstract
- Each line: `LABEL<tab>sentence`

## Citation
```bibtex
@inproceedings{dernoncourt2017pubmed,
  title={PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts},
  author={Dernoncourt, Franck and Lee, Ji Young},
  booktitle={IJCNLP},
  year={2017}
}
```