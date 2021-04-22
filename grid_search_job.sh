#!/bin/bash -f 
#SBATCH --job-name=grid
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=05:00:00
#SBATCH --output=/fred/oz002/amandlik/RFI_full_overvations_phone_call/slurm_%A_%a_grid.stdout 
#SBATCH --error=/fred/oz002/amandlik/RFI_full_overvations_phone_call/slurm_%A_%a_grid.stderr 
module purge
module load anaconda3/5.1.0
eval "$(/fred/oz002/amandlik/ana/bin/conda shell.bash hook)"
source activate ABC_II

python gridsearch.py





