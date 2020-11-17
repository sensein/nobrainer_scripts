#!/bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH -o kwyk_train.out

set -e
source ~/.bashrc
conda activate nobrainer

python3 run_kwyk_mirror_trainbatch.py 128 concrete

echo 'Finished'
