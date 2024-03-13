#!/bin/bash

echo "Hello, world!"
cd /root/
ssh-keygen
cat .ssh/id_rsa.pub
git clone https://github.com/dwarkeshsp/SAE_MoE.git
cd SAE_MoE
pip install -r requirements.txt
cd utils
python load_model.py

