#!/bin/bash 
#SBATCH --account=def-cbelling
#SBATCH --job-name=baseline_s_4
#SBATCH --cpus-per-task=12
#SBATCH --time=0-48:00
#SBATCH --mem=80G
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

#python  /home/cheryl16/projects/def-durandau/RL-Chemist/Eval_reach_vel.py UR10eReachObject-v0 UR10eReachFixed-v4
#parallel -j 5 python /home/cheryl16/projects/def-durandau/RL-Chemist/Train_reach_wandb.py --env_name 'UR10eReachObject-v0' --group 'experiment_s_3' --num_envs 3 --seed ::: {1..3}
parallel -j 5 python /home/cheryl16/projects/def-durandau/RL-Chemist/Train_reach_wandb.py --env_name 'UR10eReachObject-v0' --group 'experiment_s_4' --num_envs 3 --seed {1} --learning_rate {2} --clip_range {3} ::: {1..3} ::: 0.00001 0.0001 0.0002 ::: 0.2 0.1 0.01
