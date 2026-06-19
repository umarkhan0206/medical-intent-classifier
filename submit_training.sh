#!/bin/bash
#SBATCH --job-name=distilbert_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err

module load python/3.13.0
source venv/bin/activate

python src/train_distilbert.py
