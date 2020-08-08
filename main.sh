#!/bin/bash
#SBATCH --job-name=pythia_gradcam
#SBATCH --output=logs/logs-%j.out
#SBATCH --error=logs/logs-%j.err
#SBATCH --gres gpu:2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --partition=short

source /nethome/mummettuguli3/anaconda2/bin/activate
#conda activate maskrcnn_benchmark_fresh_2
conda activate maskrcnn_benchmark_local_2
python main.py
