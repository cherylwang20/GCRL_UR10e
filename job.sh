#!/bin/bash 
#SBATCH --account=def-durandau
#SBATCH --cpus-per-task=4
#SBATCH --time=0-0:15
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=huiyi.wang@mail.mcgill.ca
#SBATCH --mail-type=ALL

export PYTHONPATH="$PYTHONPATH:/home/cheryl16/projects/def-durandau/RL-Chemist"

cd /home/cheryl16/projects/def-durandau/RL-Chemist

module load StdEnv/2023
module load gcc opencv cuda/12.2 python/3.10 mpi4py mujoco/3.1.6

source /home/cheryl16/py310/bin/activate

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python /home/cheryl16/projects/def-durandau/RL-Chemist/Train_reach_vel.py