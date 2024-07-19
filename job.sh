#!/bin/bash

# Specify the partition of the cluster to run on (Typically TrixieMain)
#SBATCH --partition=TrixieMain
# Add your project account code using -A or --account
#SBATCH --account AI4D-CORE-148
# Specify the time allocated to the job. Max 12 hours on TrixieMain queue.
#SBATCH --time=01:00:00
# Request GPUs for the job. In this case 4 GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name=visual_training_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G        # Adjust memory request as needed
#SBATCH --time=02:00:00  # Adjust time limit as needed
#SBATCH --exclusive      # Request exclusive node access
#SBATCH --mail-type=ALL   # Send email on begin, end, and fail
#SBATCH --mail-user=huiyi.wang@nrc-cnrc.gc.ca
# Print out the hostname that the jobs is running on
hostname


# unset PIP_CONFIG_FILE; unset PYTHONPATH; 
# Activate the conda pytorch environment created in step 1
conda activate mujoco_env
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
# Launch our test pytorch python files

#python Train_reach_CNN.py
srun --ntasks=1  python Train_reach_vel.py
#xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python Train_reach_CNN.pysource