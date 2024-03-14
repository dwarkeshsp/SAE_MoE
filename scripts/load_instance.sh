#!/bin/bash

echo "Hello, world!"
cd /root/
ssh-keygen
huggingface-cli login --token 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
git config --global user.name "Dwarkesh Patel"
git config --global user.email "dwarkesh.sanjay.patel@gmail.com"
git clone https://github.com/dwarkeshsp/SAE_MoE.git
cd SAE_MoE
pip install -r requirements.txt
cd utils
python load_model.py

