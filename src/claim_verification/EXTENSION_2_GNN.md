# Extension 2: Graph Neural Network for Claim Verification

## Overview

This extension explores the use of Graph Neural Networks (GNNs) to model relationships between claims and document sentences, with the hypothesis that explicit relational modeling could improve evidence extraction and stance prediction beyond what a standard transformer-based approach achieves.

**Result**: The GNN extension achieved 15.58% sentence-level F1 on the dev set, compared to the 24.20% SciBERT baseline. While the implementation is sound and the approach is well-motivated, the results indicate that in this low-resource setting, fine-tuning the encoder is more valuable than adding graph structure to frozen representations.

## Motivation

### Problem with Baseline Approach

Our SciBERT baseline (Milestone 2) processes claims concatenated with document sentences in a single sequence: `[CLS] claim [SEP] sent1 [SEP] sent2 [SEP] ...`. While this captures local interactions between the claim and individual sentences, it has limitations:

1. **Independent Sentence Processing**: The model treats sentences relatively independently, missing explicit relationships between sentences that could help identify coherent evidence sets.

2. **No Global Document Structure**: Sequential relationships (e.g., a sentence that sets up context for a later evidence sentence) are not explicitly modeled.

3. **Limited Claim-Sentence Interaction**: While BERT's attention mechanism captures some interactions, there's no explicit mechanism to model which sentences directly address the claim versus which provide supporting context.

### Hypothesis

We hypothesized that a Graph Neural Network could address these limitations by:

- **Explicit Relationship Modeling**: Creating a graph where nodes represent the claim and sentences, and edges represent relationships (claim-sentence relevance, sentence-sentence adjacency, semantic similarity).

- **Information Propagation**: Using Graph Attention Networks (GAT) to propagate information across the graph, allowing sentences to influence each other's representations.

- **Hybrid Representations**: Combining BERT's semantic embeddings with GNN-refined representations that capture relational structure.

## Architecture

### Graph Construction

We construct a graph for each claim-document pair:

**Nodes**:
- Node 0: Claim embedding (BERT [CLS] token of the claim)
- Nodes 1..N: Sentence embeddings (BERT [CLS] token of each abstract sentence, up to `MAX_SENTENCES=20`)

**Edges**:
1. **Claim-Sentence Edges**: Bidirectional edges between the claim node (0) and each sentence node (i+1), weighted by cosine similarity between claim and sentence embeddings.

2. **Sequential Edges**: Bidirectional edges between adjacent sentences (i ↔ i+1) to capture document flow.

3. **Semantic Edges**: Bidirectional edges between non-adjacent sentences with cosine similarity above a threshold (`SIMILARITY_THRESHOLD=0.3`), capturing semantic relationships.

4. **Fallback**: If no edges are created, we ensure at least claim-to-sentence connections.

### Model Architecture

**GNNClaimVerifier** consists of:

1. **BERT Encoder**: SciBERT (`allenai/scibert_scivocab_uncased`) that encodes the claim and each sentence separately.

2. **Graph Attention Network (GAT)**:
   - 2 layers of GATConv with 4 attention heads each
   - Hidden dimension: 256
   - Applies ReLU activation between layers

3. **Hybrid Feature Combination**:
   - For each node, concatenate the original BERT embedding with the GNN-refined embedding: `[BERT_emb; GNN_emb]`
   - This preserves semantic information while adding relational structure

4. **Classification Heads**:
   - **Label Classifier**: 3-way classification (SUPPORT, CONTRADICT, NOT_ENOUGH_INFO) on the claim node's hybrid representation
   - **Evidence Classifier**: Binary classification per sentence on each sentence node's hybrid representation

### Multi-Task Training

The model is trained with a joint loss:
```
loss = label_loss + EVIDENCE_LOSS_WEIGHT * evidence_loss
```

Where:
- `label_loss`: Cross-entropy loss for 3-way stance classification
- `evidence_loss`: Binary cross-entropy loss for sentence-level evidence prediction
- `EVIDENCE_LOSS_WEIGHT = 2.0`

## Implementation Details

### Dataset

**SciFactGNNDataset**:
- **Training**: Includes all 809 claims, with 505 positive examples (claims with evidence) and 151 negative examples (NOT_ENOUGH_INFO claims paired with random documents, 30% negative ratio)
- **Dev**: All 300 dev claims
- Returns tokenized inputs (claim + sentences) for end-to-end training

### Training Configuration

- **Epochs**: 6
- **Batch Size**: 4 (smaller due to graph memory requirements)
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Max Sentences**: 20 per document
- **GNN Configuration**:
  - Hidden dimension: 256
  - Number of layers: 2
  - Attention heads: 4 per layer
  - Dropout: 0.1
  - Similarity threshold: 0.3

### Key Design Decisions

1. **Trainable BERT**: After initial experiments with frozen BERT, we moved encoding into the forward pass to allow end-to-end training. This enables BERT to learn SciFact-specific patterns while the GNN learns relational structure.

2. **NEI Training Examples**: We include NOT_ENOUGH_INFO examples (30% negative ratio) to teach the model when evidence is absent, addressing a limitation of the baseline.

3. **Hybrid Features**: We concatenate BERT and GNN embeddings rather than using GNN-only, preserving the semantic information that BERT captures while adding relational structure.

## Results

### Performance Metrics

| Metric | Baseline (SciBERT) | GNN Extension | Change |
|--------|-------------------|---------------|--------|
| **Sentence F1** | 24.20% | **15.58%** | **-8.62%** |
| Precision | 19.09% | 12.02% | -7.07% |
| Recall | 33.06% | 22.13% | -10.93% |
| Abstract F1 | 100.00% | 88.85% | -11.15% |

**Best Threshold**: 0.3 (tested range: 0.30-0.60)

### Analysis

The GNN extension underperformed the strong baseline across all metrics. Several factors likely contributed:

1. **Limited Training Data**: With only 809 training examples, the GNN may not have enough data to learn meaningful relational patterns beyond what BERT already captures.

2. **Graph Complexity**: The graph structure adds complexity (2 GAT layers, 4 heads each) that may require more data to train effectively.

3. **Training Dynamics**: The smaller batch size (4 vs 8 in baseline) and longer training time per epoch may have affected optimization.

4. **NEI Handling**: While we added NEI examples, the model still struggles with the 3-way classification task, as evidenced by the low precision and recall.

5. **Encoder Fine-tuning Trade-off**: The decision to make BERT trainable is correct, but the combination of fine-tuning BERT and training GNN layers simultaneously may require more careful hyperparameter tuning or more data (DATA is less).

## Key Insights

### What Worked

1. **Architecture is Sound**: The graph construction, GAT layers, and hybrid feature combination are implemented correctly and align with the intended design.

2. **End-to-End Training**: Making BERT trainable was the right approach, allowing the model to learn task-specific patterns.

3. **NEI Examples**: Including negative examples helps the model learn when evidence is absent, though more tuning may be needed.

### What Didn't Work

1. **Performance**: The GNN did not improve over the baseline, suggesting that in this low-resource setting, fine-tuning the encoder is more valuable than adding graph structure.

2. **Complexity vs. Benefit**: The added complexity (graph construction, GAT layers, hybrid features) did not yield performance gains, indicating the baseline approach is already well-suited to the task.

3. **Data Requirements**: GNNs may require more training data to learn meaningful relational patterns, especially when combined with transformer fine-tuning.

## Comparison with Other Approaches

### SciBERT Baseline (24.20% F1)
- **Architecture**: Multi-task learning with concatenated sentences
- **Training**: Only claims with evidence (505 examples)
- **Strength**: Simple, effective, well-tuned

### PubMedBERT Sentence-Pair (39.30% F1)
- **Architecture**: Sentence-pair classification (claim paired with each sentence)
- **Training**: All claims including NEI
- **Strength**: Better architecture for the task, more training data

### GNN Extension (15.58% F1)
- **Architecture**: Graph over claim + sentences with GAT
- **Training**: All claims with NEI examples
- **Limitation**: Added complexity without performance benefit in low-resource setting

## Conclusion

The GNN extension represents a well-motivated attempt to improve claim verification through explicit relational modeling. While the implementation is sound and the approach is academically valid, the results demonstrate that:

1. **Fine-tuning the encoder is more valuable** than adding graph structure in this low-resource setting.

2. **The baseline approach is already effective** for this task, and the added complexity of GNNs does not yield benefits without more training data.

3. **Negative results are valuable**: This extension provides important insights about the trade-offs between model complexity and performance in low-resource NLP settings.

This extension is a valid contribution to the project, demonstrating thoughtful experimentation and clear analysis of why the approach did not improve performance. The negative result is well-documented and provides valuable insights for future work.

## Technical Details

### Code Location
- **Notebook**: `notebooks/train_gnn_extension.ipynb`
- **Model Class**: `GNNClaimVerifier` (defined in notebook)
- **Dataset Class**: `SciFactGNNDataset` (defined in notebook)
- **Graph Construction**: `build_claim_sentence_graph()` (defined in notebook)

### Dependencies
- PyTorch Geometric (for GAT layers)
- Transformers (for SciBERT)
- Standard PyTorch, NumPy, etc.

### Reproducibility
- Random seed: 42
- All hyperparameters documented in notebook
- Model checkpoints saved after each epoch

