## Strong Baseline: Section Classification

This file documents the strong baseline for **Component 2: Section Classification**, as required for Milestone 2.

---

## Model Description

For our strong baseline, we fine-tuned a pre-trained **SciBERT** model (`allenai/scibert_scivocab_uncased`).

We trained this model on the **PubMed 200K RCT dataset** to classify individual sentences from medical abstracts into one of five rhetorical sections:

* BACKGROUND
* OBJECTIVE
* METHODS
* RESULTS
* CONCLUSIONS

We fine-tuned the model for **3 epochs** using the Hugging Face Trainer. The Trainer was set to `load_best_model_at_end=True`, so the final model we evaluated was from Epoch 1, which had the lowest validation loss. 

## Performance

The model was evaluated on the `dev.txt` dataset (**30,212 sentences**).

* **Overall Accuracy:** 88.15%
* **Macro-F1 Score:** 82.40%

### Detailed Classification Report

Here is the full report showing the performance for each class:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| BACKGROUND | 0.6919 | 0.8202 | 0.7506 | 3449 |
| OBJECTIVE | 0.8241 | 0.5324 | 0.6469 | 2376 |
| **METHODS** | **0.9357** | **0.9556** | **0.9456** | 9964 |
| **RESULTS** | **0.9298** | **0.9261** | **0.9280** | 9841 |
| CONCLUSIONS | 0.8462 | 0.8514 | 0.8488 | 4582 |
| | | | | |
| **Accuracy** | | | **0.8815** | 30212 |
| **Macro Avg** | 0.8455 | 0.8172 | **0.8240** | 30212 |
| **Weighted Avg** | 0.8836 | 0.8815 | 0.8794 | 30212 |



## How to Reproduce

1. Make sure all libraries from `requirements.txt` are installed.
2. Open the **notebooks/train_scibert_section_classification.ipynb** notebook.
3. **Use a GPU!** \
In Colab, make sure you have a T4 GPU enabled (Runtime > Change runtime type).
4. Run all the cells in order from top to bottom.
5. The final trained model is saved in the `models/scibert_section_classifier/` directory.