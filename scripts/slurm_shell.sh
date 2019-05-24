#!/bin/bash
#SBATCH -A ac5616p
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta16:1
#SBATCH --time=1:00:00
#SBATCH --mail-user=brianleepollack@gmail.com
#SBATCH --output=/pylon5/ac5616p/bpollack/mre_slurm/test_output.out

set -x
echo "$@"
source /pghbio/dbmi/batmanlab/bpollack/anaconda3/etc/profile.d/conda.sh
conda activate new_mre
"$@"

python mre/train_model_full.py --num_epochs=10 --model_version='slurm_test'
