#!/bin/bash 
#SBATCH --account=def-cbelling
#SBATCH --job-name=baseline_rbf_train
#SBATCH --cpus-per-task=12
#SBATCH --time=0-72:00
#SBATCH --mem=60G
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

wandb offline

#python  /home/cheryl16/projects/def-durandau/RL-Chemist/Eval_reach_vel.py
parallel -j 5 python /home/cheryl16/projects/def-durandau/RL-Chemist/Train_reach_wandb.py --env_name 'UR10eReachFixed-v4' --group 'baseline_experiment_4_rbf' --num_envs 2 --seed ::: {1..10}
