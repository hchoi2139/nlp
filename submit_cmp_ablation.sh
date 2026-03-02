#!/bin/bash
#SBATCH --job-name=pcl_cmp_abl
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/cmp_abl_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/cmp_abl_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00

echo "--- Starting Ablation Studies on A100 ---"

uv run python eval/compare_ablation.py

echo "--- Ablation Studies Complete ---"