#!/bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -n 4
#SBATCH --gres=gpu:tesla-v100:2
#SBATCH -p gablab
#SBATCH --mem=32G
#SBATCH -o /om2/user/hodaraja/kwyk/nobrainer_scripts/outputs/slurm_2gpu_%j.out

set -e
unset XDG_RUNTIME_DIR


IMAGE="/om2/user/hodaraja/containers/tf2_nobrainer_0.sif"

singularity exec \
--nv \
-B /om/user/satra/kwyk/tfrecords/ \
-B /om2/user/hodaraja \
"$IMAGE" \
python3 run_kwyk_mirror_trainbatch.py 128

echo 'Finished'
