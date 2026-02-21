#!/bin/bash
#SBATCH --job-name=pcl_optuna
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/optuna_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/optuna_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

echo "--- Starting Optuna Hyperparameter Sweep on A100 ---"

uv run python src/tune.py

echo "--- Optuna Sweep Complete ---"