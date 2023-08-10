#!/bin/bash
#SBATCH -c 6
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=96G
#SBATCH -C gmem48
#SBATCH -p gpu
#SBATCH --time=0-70:00:00
module load cuda/11.7 cudnn/10-74
source activate transgeo
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
source run_CVUSA2.sh