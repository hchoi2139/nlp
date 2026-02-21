#!/bin/bash
#SBATCH --job-name=PCL_FineTune
#SBATCH --partition=a100            # Target the 80GB A100 partition
#SBATCH --gres=gpu:1                # Request 1 GPU from that partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/vol/bitbucket/hc1721/nlp_scratch/logs/train_%j.out
#SBATCH --error=/vol/bitbucket/hc1721/nlp_scratch/logs/train_%j.err

# Point to your fast scratch storage for model weights
export HF_HOME=/vol/bitbucket/hc1721/nlp_scratch/huggingface
export UV_CACHE_DIR=/vol/bitbucket/hc1721/nlp_scratch/uv_cache

# Navigate to where your code actually lives
cd /homes/hc1721/nlp

# Run training
uv run python src/main.py