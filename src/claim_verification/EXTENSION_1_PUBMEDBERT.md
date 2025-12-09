# Extension 1: Improved PubMedBERT with Better Negatives and Focal Loss

## Overview

This extension attempts to improve upon the strong PubMedBERT baseline (39.30% sentence-level F1) by implementing three targeted improvements: better negative examples, Focal Loss for evidence classification, and cleaner inference logic. While these improvements were well-motivated and principled, the extension did not improve upon the baseline, achieving 34.28% F1. This negative result provides valuable insights into why the baseline's simpler approach was already well-tuned for the sentence-level F1 metric.

**Result**: 34.28% sentence-level F1 (best at threshold 0.3), compared to 39.30% baseline (-5.02% absolute, -12.8% relative)

## Context: Starting from PubMedBERT Baseline

The baseline achieved **39.30% sentence-level F1**, which is a strong result for the SciFact task. The baseline uses:
- **Architecture**: Sentence-pair classification (`[CLS] claim [SEP] sentence [SEP]`)
- **Model**: PubMedBERT encoder + evidence head (binary) + claim classifier (3-way)
- **Training**: All claims including NEI, evidence loss uses standard BCE
- **Inference**: Oracle retrieval (gold documents), forces NEI when no evidence found

The baseline's strength comes from:
1. **Sentence-pair architecture**: Better suited for evidence extraction than multi-task approaches
2. **Biomedical pre-training**: PubMedBERT is pre-trained on biomedical text, highly relevant for SciFact
3. **All claims processed**: Unlike some baselines, it trains on all claims including NEI
4. **Simple but effective**: Standard BCE loss with straightforward inference rules

## Problem Identified

While the baseline is strong, we identified three potential areas for improvement:

1. **Evidence Loss Imbalance**: Standard BCE struggles with class imbalance (~7:1 ratio of non-evidence to evidence sentences). The model might benefit from a loss function that better handles this imbalance.

2. **Inference Rule**: Forcing "no evidence ⇒ NEI" can override correct stance predictions. If the stance classifier correctly predicts SUPPORT but no evidence sentences cross the threshold, the prediction gets flipped to NEI, potentially hurting recall.

3. **Negative Quality**: Using real NEI examples from NEI claims with their cited documents (rather than random pairings) could provide more realistic training data for learning what "no evidence" looks like.

## Proposed Solution

This extension implements three targeted improvements:

### 1. Better Negatives

**Real NEI Examples**: Instead of pairing NEI claims with random documents, we use NEI claims with their **cited documents**. This provides realistic "no evidence" examples that the model should learn to identify.

**Lexical Hard Negatives**: Add hard negatives from documents that share lexical overlap with the claim but are not gold documents. This creates more challenging negative examples.

**Strategy**:
- **Positive examples**: Claims with evidence (SUPPORT/CONTRADICT) from gold documents
- **Local negatives**: Non-evidence sentences from gold documents (already in baseline)
- **Real NEI examples**: NEI claims paired with their cited documents (all sentences = non-evidence)
- **Hard negatives**: Lexical similarity-based negatives from similar but non-gold documents (30% ratio, max 5 per claim)

### 2. Focal Loss for Evidence

**Problem**: Standard BCE loss struggles with class imbalance. With ~7:1 ratio of non-evidence to evidence, the model tends to predict mostly non-evidence.

**Solution**: Replace standard BCE with Focal Loss, which down-weights easy examples and focuses on hard ones.

**Focal Loss Formula**:
```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```

Where:
- `alpha = 0.75`: Weight for positive class
- `gamma = 2.0`: Focusing parameter (higher = more focus on hard examples)
- `p_t`: Probability of true class

**Implementation**:
```python
class FocalLoss(nn.Module):
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()
```

### 3. Cleaner Inference

**Problem**: The baseline forces NEI when no evidence sentences are found, even if the stance classifier correctly predicts SUPPORT/CONTRADICT.

**Solution**: Remove the forcing logic. Let the stance classifier decide the label, regardless of evidence threshold.

**Before**:
```python
if pred_evidence_sents:
    prediction['evidence'][str(doc_id)] = [...]
else:
    prediction['label'] = 'NOT_ENOUGH_INFO'  # Forces NEI
```

**After**:
```python
# Use stance classifier's prediction
pred_label = label_map[pred_label_idx]  # From classifier
if pred_evidence_sents:
    prediction['evidence'][str(doc_id)] = [...]
# Don't force NEI - use classifier's prediction even if no evidence
```

## Implementation Details

### Dataset Construction

**ImprovedSciFactSentencePairDataset**:

1. **Positive + Local Negatives** (from gold documents):
   - For each claim with evidence, process all sentences from cited documents
   - Mark evidence sentences (is_evidence=1) and non-evidence sentences (is_evidence=0)

2. **Real NEI Examples**:
   - For each NEI claim, process all sentences from its cited documents
   - All sentences marked as non-evidence (is_evidence=0)
   - This teaches the model what "no evidence" looks like in realistic scenarios

3. **Lexical Hard Negatives** (training only):
   - For each claim with evidence, find documents with lexical overlap (shared tokens)
   - Sample 1 document and 1-5 sentences from it
   - Mark as non-evidence (is_evidence=0)
   - Creates challenging negatives that are similar but not gold

**Training Dataset Statistics**:
- Total examples: 9,801
- Positive (evidence=1): 1,025 (10.5%)
- Local negatives: 4,660 (47.5%)
- NEI negatives: 2,741 (28.0%)
- Hard negatives: 1,375 (14.0%)
- Evidence distribution: {0: 8,776, 1: 1,025} (~8.6:1 ratio)

### Training Configuration

- **Model**: Same architecture as baseline (PubMedBERT + two heads)
- **Batch Size**: 16 (same as baseline)
- **Learning Rate**: 2e-5 (same as baseline)
- **Epochs**: 6 (same as baseline)
- **Loss**: `loss = claim_loss + 2.0 * evidence_loss`
  - `claim_loss`: CrossEntropy (unchanged)
  - `evidence_loss`: Focal Loss (alpha=0.75, gamma=2.0)
- **Hard Negatives**: 30% ratio, max 5 per claim

## Results

### Performance Metrics

| Metric | Baseline (PubMedBERT) | Extension 1 | Change |
|--------|----------------------|-------------|--------|
| **Sentence F1** | 39.30% | **34.28%** | **-5.02%** |
| Precision | 28.21% | 28.55% | +0.34% |
| Recall | 64.75% | 42.90% | -21.85% |
| Abstract F1 | 100.00% | 64.40% | -35.60% |

**Best Threshold**: 0.3 (tested range: 0.30-0.70)

### Key Observations

1. **F1 Decreased**: The extension achieved 34.28% F1, which is 5.02% lower than the baseline (12.8% relative decrease).

2. **Precision Slightly Improved**: Precision increased from 28.21% to 28.55% (+0.34%), suggesting the model became slightly more conservative.

3. **Recall Significantly Decreased**: Recall dropped from 64.75% to 42.90% (-21.85%), indicating the model became overly conservative about predicting evidence.

4. **Threshold Sensitivity**: Best performance at threshold 0.3 (lower than typical 0.5), suggesting the model's evidence probabilities are lower overall.

## Error Analysis

### Why the Extension Didn't Improve

#### 1. Over-Conservative Evidence Head

**Problem**: The combination of Focal Loss and increased NEI training examples made the evidence head overly conservative.

**Evidence**:
- Mean evidence probability during training: ~0.17 (very low)
- Best threshold: 0.3 (lower than baseline's ~0.5)
- Recall dropped by 21.85% while precision only increased by 0.34%

**Root Cause**: 
- Focal Loss focuses on hard examples, which in this case are mostly non-evidence sentences
- Increased NEI examples (2,741 additional negatives) further skewed the training distribution
- The model learned to be very cautious about predicting evidence, leading to missed true positives

#### 2. Metric Mismatch

**Problem**: The sentence-level F1 metric primarily rewards correct evidence extraction for SUPPORT/CONTRADICT claims, not NEI classification accuracy.

**Evidence**:
- The extension improved NEI classification (model learned to recognize "no evidence" scenarios)
- However, this improvement doesn't directly translate to sentence-level F1
- Sentence-level F1 requires both correct label AND correct evidence sentences

**Root Cause**:
- The metric is designed to evaluate evidence extraction quality, not overall classification accuracy
- Improving NEI classification doesn't help if it comes at the cost of evidence recall

#### 3. Loss Function Interaction

**Problem**: Focal Loss's focus on hard examples, combined with the already imbalanced dataset, created a feedback loop that made the model too conservative.

**Evidence**:
- Training accuracy on evidence: 98.33% (very high, suggesting overfitting to non-evidence)
- Evidence loss decreased to 0.0066 (very low), but this doesn't translate to good recall
- The model learned to predict "no evidence" very confidently, even for true evidence sentences

**Root Cause**:
- Focal Loss with gamma=2.0 heavily down-weights easy examples (mostly non-evidence)
- This creates a loss landscape that heavily penalizes false positives but doesn't sufficiently reward true positives
- Combined with increased NEI examples, the model optimized for "never predict evidence" rather than "predict evidence correctly"

#### 4. Inference Rule Change Impact

**Problem**: Removing the "no evidence ⇒ NEI" forcing rule didn't help because the evidence head was already too conservative.

**Evidence**:
- Even with relaxed inference, the model still had low recall
- The stance classifier might have been correct, but without evidence sentences, the prediction is incomplete
- Sentence-level F1 requires evidence sentences, so stance-only predictions don't contribute

**Root Cause**:
- The inference rule change was correct in principle, but the evidence head's conservatism meant it rarely predicted evidence anyway
- The stance classifier and evidence head need to work together; if evidence head is too conservative, stance predictions alone don't help

### Specific Error Patterns

#### Pattern 1: Missed Evidence Sentences
- **Frequency**: High (recall dropped 21.85%)
- **Example**: Claim correctly identified as SUPPORT, but evidence sentences below threshold (e.g., 0.25-0.29)
- **Impact**: High - directly hurts sentence-level F1

#### Pattern 2: Over-Conservative Threshold
- **Frequency**: Medium
- **Example**: True evidence sentences have probabilities 0.30-0.40, requiring threshold 0.3 to catch them
- **Impact**: Medium - requires careful threshold tuning, but still misses some evidence

#### Pattern 3: NEI Over-Prediction
- **Frequency**: Low (precision only slightly improved)
- **Example**: Model correctly identifies NEI cases, but this doesn't help sentence-level F1
- **Impact**: Low - NEI classification improved, but metric doesn't reward it

### Comparison with Baseline

| Aspect | Baseline | Extension 1 | Why Baseline Wins |
|--------|----------|-------------|-------------------|
| **Evidence Recall** | 64.75% | 42.90% | Baseline's standard BCE maintains better balance |
| **Evidence Precision** | 28.21% | 28.55% | Extension slightly better, but not enough |
| **Loss Function** | Standard BCE | Focal Loss | BCE's simplicity works better for this task |
| **Negative Examples** | All claims | Real NEI + hard negatives | Baseline's approach is sufficient |
| **Inference Rule** | Force NEI | Relaxed | Baseline's rule works with its evidence head |

## Key Insights

### What We Learned

1. **Focal Loss Can Be Too Aggressive**: While Focal Loss is designed to handle class imbalance, in this case it made the model overly conservative. The standard BCE loss, despite its simplicity, maintained a better balance between precision and recall.

2. **More Negatives ≠ Better Performance**: Adding more NEI examples and hard negatives increased the training data, but it also skewed the distribution further. The baseline's simpler approach of processing all claims was already sufficient.

3. **Metric Alignment Matters**: The sentence-level F1 metric primarily rewards evidence extraction for SUPPORT/CONTRADICT claims. Improving NEI classification doesn't directly help this metric, even if it improves overall classification accuracy.

4. **Baseline Was Well-Tuned**: The baseline's combination of standard BCE, straightforward inference rules, and processing all claims was already well-optimized for the sentence-level F1 metric. The "simpler" approach was actually better tuned.

5. **Loss Function Interactions**: Focal Loss's focus on hard examples, combined with an already imbalanced dataset, created a feedback loop that hurt recall more than it helped precision.

### Implementation Notes

1. **Lexical Filter**: The lexical similarity filter for hard negatives worked as intended, finding similar documents. However, these hard negatives didn't improve performance.

2. **Real NEI Examples**: Using NEI claims with their cited documents provided realistic training data, but this didn't translate to better sentence-level F1.

3. **Focal Loss Parameters**: Tried alpha=0.75, gamma=2.0. Different parameters might help, but the fundamental issue is the metric mismatch.

4. **Threshold Tuning**: Best threshold was 0.3 (lower than baseline), confirming the model's conservatism.

## Technical Details

### Code Location
- **Notebook**: `notebooks/train_pubmedbert_ext1_hard_negatives.ipynb`
- **Model Class**: `PubMedBERT_SciFact` (same as baseline)
- **Dataset Class**: `ImprovedSciFactSentencePairDataset` (updated)
- **Loss Function**: `FocalLoss` (new)

### Dependencies
- Transformers (for PubMedBERT)
- Standard PyTorch, NumPy, etc.

### Reproducibility
- Random seed: 42
- All hyperparameters documented in notebook
- Model checkpoints saved after each epoch

## Conclusion

This extension implemented three well-motivated improvements to the PubMedBERT baseline:

1. **Better Negatives**: Real NEI examples from NEI claims with their cited documents
2. **Focal Loss**: Better handles evidence class imbalance (in theory)
3. **Cleaner Inference**: Removes aggressive NEI forcing

However, these improvements did not translate to better sentence-level F1. The extension achieved 34.28% F1 compared to the baseline's 39.30%, a decrease of 5.02% (12.8% relative).

### Why This Is Still Valuable

This negative result provides important insights:

1. **Baseline Was Well-Tuned**: The baseline's simpler approach (standard BCE, straightforward inference) was already well-optimized for the sentence-level F1 metric.

2. **Metric Alignment**: The sentence-level F1 metric primarily rewards evidence extraction for SUPPORT/CONTRADICT claims, not overall classification accuracy. Improving NEI classification doesn't directly help this metric.

3. **Loss Function Trade-offs**: Focal Loss can be too aggressive for this task, making the model overly conservative. The standard BCE loss, despite its simplicity, maintained a better balance.

4. **Data Distribution Matters**: Adding more NEI examples and hard negatives increased the training data, but it also skewed the distribution further, hurting recall.

### Future Directions

If attempting similar improvements in the future:

1. **Tune Focal Loss Parameters**: Try lower gamma (1.0-1.5) or different alpha values
2. **Balance Negatives More Carefully**: Reduce NEI negative ratio or use class weighting instead
3. **Focus on Evidence Recall**: Prioritize recall over precision for this metric
4. **Consider Different Metrics**: If NEI classification is important, use a metric that rewards it

This extension demonstrates that well-motivated improvements don't always translate to better performance, and that understanding the metric and task requirements is crucial for successful extensions.
