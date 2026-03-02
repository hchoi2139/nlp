#!/bin/bash
#SBATCH --job-name=pcl_ablate
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/ablation_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/ablation_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00

echo "--- Starting Ablation Studies on A100 ---"

uv run python eval/run_ablation.py

echo "--- Ablation Studies Complete ---"