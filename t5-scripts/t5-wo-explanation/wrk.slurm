#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL

bash explaint5.sh