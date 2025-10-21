#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# Job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --partition=instruction    # Use the instruction partition
#SBATCH --nodes=1   # Number of nodes to use
#SBATCH --ntasks-per-node=8   # Use 8 processor cores per node 
#SBATCH --time=0-5:0:0   # Walltime limit (DD-HH:MM:SS)
#SBATCH --mem=32G   # Maximum memory per node
#SBATCH --gres=gpu:1   # Required GPU hardware
#SBATCH -A f2025.coms.6250.02   # Slurm account to use for the job
#SBATCH --job-name="multi-modality-clip"   # Job name to display in squeue
#SBATCH --mail-user=dhawal04@iastate.edu   # Email address
#SBATCH --mail-type=BEGIN,END,FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source "/work/classtmp/dhawal04/.venv/bin/activate"

BASE="/work/classtmp/dhawal04/Advance AI/engagement-predictor/finetune_CLIP"
cd "$BASE/"

python finetune_these_nuts.py