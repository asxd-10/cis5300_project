#!/bin/bash
# --- Conda Environment and Dependencies ---
conda create -n scifact python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scifact
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# PubMed 200K RCT Download
cd data/pubmed_rct
wget -q https://github.com/Franck-Dernoncourt/pubmed-rct/raw/master/PubMed_20k_RCT_numbers_replaced_with_at_sign/train.txt
wget -q https://github.com/Franck-Dernoncourt/pubmed-rct/raw/master/PubMed_20k_RCT_numbers_replaced_with_at_sign/dev.txt
wget -q https://github.com/Franck-Dernoncourt/pubmed-rct/raw/master/PubMed_20k_RCT_numbers_replaced_with_at_sign/test.txt
cd ../..
echo "Setup complete"