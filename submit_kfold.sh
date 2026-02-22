#!/bin/bash
#SBATCH --job-name=pcl_kfold
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/kfold_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/kfold_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

echo "--- Starting 5-Fold Cross Validation on A100 ---"

# Run the K-Fold training pipeline
uv run python src/train_kfold.py

echo "--- K-Fold Training Complete ---"