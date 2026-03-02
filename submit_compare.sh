#!/bin/bash
#SBATCH --job-name=pcl_compare
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/compare_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/compare_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

echo "--- Starting Baseline Comparison & Error Analysis on A100 ---"

uv run python eval/compare_models.py

echo "--- Error Analysis Complete ---"