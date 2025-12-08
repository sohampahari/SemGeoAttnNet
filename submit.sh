#!/bin/bash 
#PBS -N test 
#PBS -q gpu 
#PBS -l select=1:ncpus=10:ngpus=1:mem=10g 
#PBS -j oe 
#PBS -V  
cd $PBS_O_WORKDIR 
source /home/soft/anaconda3/etc/profile.d/conda.sh  
conda init  
conda activate  
conda activate vsalenv  
# Replace pytorch-gpu with your environment name
module load cuda 
python3 ./data_aug.py 
# Replace data_aug.py with your script name

# GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-51ff52a6-9be1-3ab9-ff88-65d3f97720a1)
# GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-5ac08ef4-9737-c367-bf80-200f205f6014)
# GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-e5935c6a-9ebe-ab83-389b-00c3faed3a38)
#   MIG 3g.40gb     Device  0: (UUID: MIG-e6b94c71-1db7-5ad2-a95a-9de8bd50bb1a)
#   MIG 3g.40gb     Device  1: (UUID: MIG-80f109f6-aa75-51f9-9909-42ad8029df93)``
# GPU 3: NVIDIA DGX Display (UUID: GPU-5253964e-8c75-b074-8289-0c34c6e41116)
# GPU 4: NVIDIA A100-SXM4-80GB (UUID: GPU-beb87e1b-7f0c-2da0-38cb-f982391a6842)
#   MIG 3g.40gb     Device  0: (UUID: MIG-79bd4bd1-7b07-5a71-aa77-3afd7b2e20dc)
#   MIG 3g.40gb     Device  1: (UUID: MIG-51c93f85-6919-5785-8f23-031ab153a183)

# CUDA_VISIBLE_DEVICES=GPU-51ff52a6-9be1-3ab9-ff88-65d3f97720a1 python train.py

# cd /media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/soham/3d-saliency && pip install diffusers==0.21.4 huggingface-hub==0.17.3 einops==0.7.0 transformers==4.34.1 -q