#!/bin/bash 
#SBATCH --account=def-cbelling
#SBATCH --job-name=pick_dis_cher
#SBATCH --cpus-per-task=32
#SBATCH --time=0-50:50
#SBATCH --array=5-9
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=huiyi.wang@mail.mcgill.ca
#SBATCH --mail-type=ALL

export PYTHONPATH="$PYTHONPATH:/home/cheryl16/projects/def-durandau/RL-Chemist"

cd /home/cheryl16/projects/def-durandau/RL-Chemist

module load StdEnv/2023
module load gcc opencv/4.9.0 cuda/12.2 python/3.10 mpi4py mujoco/3.1.6

source /home/cheryl16/py310/bin/activate

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

wandb offline

######Sim2Real
#python training/Train_reach.py --env_name 'UR10eReach1C-v1' --group 'Reach_4C_dt20' --num_envs 4 --learning_rate 0.0003 --clip_range 0.1 --seed=$SLURM_ARRAY_TASK_ID --channel_num 4 --fs 20

python training/Train_reach.py --env_name 'UR10ePickPlace-v0' --group 'Pick_dis_cher' --num_envs 4 --learning_rate 0.0003 --clip_range 0.1 --seed=$SLURM_ARRAY_TASK_ID --channel_num 4 --fs 20
