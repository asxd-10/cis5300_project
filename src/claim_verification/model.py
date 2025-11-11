"""
SciBERT-based claim verification model.
Two-stage approach:
1. Retrieve documents (BM25 or oracle)
2. Joint evidence extraction + label prediction
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ClaimVerifier(nn.Module):
    """
    SciBERT model for claim verification.
    Input: [CLS] claim [SEP] sent1 [SEP] sent2 [SEP] ...
    Output: 
        - Label logits (3 classes: SUPPORT, CONTRADICT, NEI)
        - Evidence logits (binary per sentence)
    """
    
    def __init__(self, model_name='allenai/scibert_scivocab_uncased', 
                 num_labels=3, max_sentences=20):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        
        # Label classifier (uses [CLS] token)
        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Evidence classifier (per sentence)
        self.evidence_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)  # Binary: is evidence or not
        )
        
        self.max_sentences = max_sentences
    
    def forward(self, input_ids, attention_mask, sentence_positions=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            sentence_positions: [batch, max_sentences] - positions of [SEP] tokens
        
        Returns:
            label_logits: [batch, 3]
            evidence_logits: [batch, max_sentences]
        """
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Label prediction from [CLS]
        cls_output = sequence_output[:, 0, :]  # [batch, hidden]
        label_logits = self.label_classifier(cls_output)  # [batch, 3]
        
        # Evidence prediction from sentence boundaries
        if sentence_positions is not None:
            batch_size = sequence_output.size(0)
            evidence_logits = []
            
            for i in range(batch_size):
                sent_reps = []
                for pos in sentence_positions[i]:
                    if pos > 0 and pos < input_ids.size(1):
                        sent_reps.append(sequence_output[i, pos, :])
                    else:
                        # Padding
                        sent_reps.append(torch.zeros_like(sequence_output[i, 0, :]))
                
                sent_reps = torch.stack(sent_reps)  # [max_sentences, hidden]
                ev_logits = self.evidence_classifier(sent_reps).squeeze(-1)  # [max_sentences]
                evidence_logits.append(ev_logits)
            
            evidence_logits = torch.stack(evidence_logits)  # [batch, max_sentences]
        else:
            evidence_logits = None
        
        return label_logits, evidence_logits


def create_input(claim: str, abstract_sentences: list, tokenizer, max_length=512):
    """
    Create model input: [CLS] claim [SEP] sent1 [SEP] sent2 [SEP] ...
    
    Returns:
        input_ids, attention_mask, sentence_positions
    """
    # Build text
    text = claim
    sentence_positions = []
    current_length = len(tokenizer.encode(claim, add_special_tokens=True))
    
    for sent in abstract_sentences:
        sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
        if current_length + len(sent_tokens) + 1 > max_length - 1:
            break
        
        text += " [SEP] " + sent
        sentence_positions.append(current_length + len(sent_tokens))
        current_length += len(sent_tokens) + 1
    
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Pad sentence positions
    max_sents = 20
    while len(sentence_positions) < max_sents:
        sentence_positions.append(0)
    sentence_positions = sentence_positions[:max_sents]
    
    return encoding['input_ids'], encoding['attention_mask'], torch.tensor([sentence_positions])
