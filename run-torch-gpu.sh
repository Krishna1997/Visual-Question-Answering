#!/bin/bash
#SBATCH --account=csb170
#SBATCH --job-name=pytorch-gpu-shared
#SBATCH --partition=large-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:k80:4
#SBATCH --time=200:00:00
#SBATCH --output=pytorch-gpu-shared.o%j.%N
#SBATCH --constraint="large_scratch"
#SBATCH --mem-455GB
                                                                                                                            module purge
module list


singularity exec --bind /oasis,/scratch --nv /share/apps/gpu/singularity/images/pytorch/pytorch-gpu.simg python main.py --model simple
echo Hello
