#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o outputs/kwyk_transfer_cce_kl_nbg_b32_cl115_%j.out
##SBATCH --exclusive


## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=nobrainer2.4
CONDA_ROOT=$HOME2/miniconda3

## Activate environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Command to run
python3 kwyk_transfer_learning_nfrz.py

echo 'Finished'
