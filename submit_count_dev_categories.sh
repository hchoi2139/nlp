#!/bin/bash
#SBATCH --job-name=pcl_compare
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/count_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/count_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

echo "--- Starting Counting Categories on Dev Set on A100 ---"

uv run python eval/count_dev_categories.py

echo "--- Error Analysis Complete ---"