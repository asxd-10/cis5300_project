"""
Verify SciFact & PubMed RCT statistics.
Run from repo root:
    python code/data_preparation/verify_data_stats.py
"""
import sys
from pathlib import Path
from collections import Counter

# ----------------------------------------------------------------------
# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]  # code/data_preparation → repo root
sys.path.insert(0, str(REPO_ROOT / "code"))

try:
    from common.data_utils import load_claims, load_corpus
except ImportError as e:
    print("ImportError: Cannot find common.data_utils")
    print("   Make sure code/common/data_utils.py exists and defines load_claims() and load_corpus()")
    sys.exit(1)
# ----------------------------------------------------------------------


def verify_scifact():
    print("\n" + "="*60)
    print("SCIFACT VERIFICATION")
    print("="*60)

    # Correct paths: inside data/scifact/data/
    base = REPO_ROOT / "data" / "scifact" / "data"
    splits = {
        "train": base / "claims_train.jsonl",
        "dev":   base / "claims_dev.jsonl",
        "test":  base / "claims_test.jsonl",
    }
    corpus_path = base / "corpus.jsonl"

    total_claims = 0
    label_counter = Counter()
    evidence_lens = []

    # Verify each split
    for name, path in splits.items():
        if not path.exists():
            print(f"  missing {name} file: {path}")
            continue

        claims = load_claims(str(path))
        total_claims += len(claims)

        for c in claims:
            # Normalize labels
            lbl = c.label
            if lbl == "SUPPORT":
                label_counter["SUPPORTS"] += 1
            elif lbl == "CONTRADICT":
                label_counter["REFUTES"] += 1
            elif lbl in ("NOT_ENOUGH_INFO", "NEI"):
                label_counter["NEI"] += 1
            else:
                label_counter[lbl] += 1

            # Count evidence sentences
            if c.evidence:
                for doc in c.evidence.values():
                    for ev in doc:
                        evidence_lens.append(len(ev.get("sentences", [])))

        print(f"  {name.upper():>5}: {len(claims)} claims")

    # Corpus
    if not corpus_path.exists():
        print(f"  missing corpus: {corpus_path}")
    else:
        corpus = load_corpus(str(corpus_path))
        abs_len = [len(d.abstract) for d in corpus.values()]
        avg_abs = sum(abs_len) / len(abs_len) if abs_len else 0
        print(f"  CORPUS: {len(corpus)} abstracts, avg {avg_abs:.1f} sentences")

    # Final summary
    print("\nSUMMARY (copy to data_scifact.md):")
    print("-"*50)
    print(f"Total claims      : {total_claims}")
    print(f"Train / Dev / Test: "
          f"{len(load_claims(str(splits['train']))) if splits['train'].exists() else 0} / "
          f"{len(load_claims(str(splits['dev']))) if splits['dev'].exists() else 0} / "
          f"{len(load_claims(str(splits['test']))) if splits['test'].exists() else 0}")
    print(f"Corpus abstracts  : {len(corpus) if 'corpus' in locals() else 0}")
    print(f"Avg abstract len  : {avg_abs:.1f} sentences" if 'avg_abs' in locals() else "")
    print("\nLabel distribution:")
    for lbl in ["SUPPORTS", "REFUTES", "NEI"]:
        cnt = label_counter.get(lbl, 0)
        pct = cnt / total_claims * 100 if total_claims else 0
        print(f"  {lbl:<9}: {cnt:>4} ({pct:4.1f}%)")
    if evidence_lens:
        print(f"Avg evidence sentences per claim: {sum(evidence_lens)/len(evidence_lens):.1f}")
    print("-"*50)
    return total_claims > 0


def verify_pubmed():
    print("\n" + "="*60)
    print("PUBMED RCT VERIFICATION")
    print("="*60)

    base = REPO_ROOT / "data" / "pubmed_rct"
    splits = {
        "train": base / "train.txt",
        "dev":   base / "dev.txt",
        "test":  base / "test.txt",
    }

    for name, path in splits.items():
        if not path.exists():
            print(f"  missing {name} file: {path}")
            continue

        abstracts = 0
        sentences = 0
        labels = Counter()

        with path.open() as f:
            for line in f:
                line = line.strip()
                if line.startswith("###"):
                    abstracts += 1
                elif line and "\t" in line:
                    lbl, _ = line.split("\t", 1)
                    labels[lbl] += 1
                    sentences += 1

        print(f"  {name.upper():>5}: {abstracts} abstracts, {sentences} sentences")
        for lbl, cnt in labels.most_common():
            pct = cnt / sentences * 100
            print(f"    {lbl:<12}: {cnt:>6} ({pct:4.1f}%)")


if __name__ == "__main__":
    print("\nDATA VERIFICATION START\n")
    scifact_ok = verify_scifact()
    verify_pubmed()
    print("\n" + "="*60)
    print("SCIFACT VERIFICATION: " + ("PASS" if scifact_ok else "FAIL"))
    print("="*60 + "\n")
