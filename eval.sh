#!/bin/bash
#SBATCH --job-name=pcl_eval
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/eval_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

echo "--- Starting Evaluation on A100 ---"
# If your environment requires loading CUDA modules:
# module load cuda/12.1 

# Run the evaluation
uv run python src/evaluate.py

echo "--- Evaluation Complete ---"