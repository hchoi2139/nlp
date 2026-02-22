#!/bin/bash
#SBATCH --job-name=pcl_infer
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/infer_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/infer_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "--- Starting Inference for BestModel ---"

# Run the submission generation pipeline
uv run python src/generate_submission.py

echo "--- Inference Complete! Check the BestModel directory. ---"