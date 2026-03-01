#!/bin/bash
#SBATCH --job-name=pcl_submit
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/kfold_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/kfold_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

echo "--- Starting generate_submission.py ---"

# Run the K-Fold training pipeline
uv run python src/generate_submission.py

echo "--- Generating Submission Complete ---"