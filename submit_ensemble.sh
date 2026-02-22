#!/bin/bash
#SBATCH --job-name=pcl_ensemble
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/ensemble_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/ensemble_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

echo "--- Starting Ensemble Evaluation on A100 ---"

uv run python src/evaluate_ensemble.py

echo "--- Ensemble Evaluation Complete ---"