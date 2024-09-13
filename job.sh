#!/bin/bash 
#SBATCH --account=def-durandau
#SBATCH --job-name=reach4_2
#SBATCH --cpus-per-task=12
#SBATCH --time=0-48:00
#SBATCH --array=1
#SBATCH --mem=108G
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

#python  /home/cheryl16/projects/def-durandau/RL-Chemist/Eval_reach_vel.py UR10eReachObject-v0 UR10eReachFixed-v4
#parallel -j 5 python /home/cheryl16/projects/def-durandau/RL-Chemist/Train_reach_wandb.py --env_name 'UR10eReachObject-v0' --group 'experiment_s_3' --num_envs 3 --seed ::: {1..3}
#parallel -j 5 python /home/cheryl16/projects/def-durandau/RL-Chemist/Train_reach_wandb.py --env_name 'UR10eReachObject-v0' --group 'experiment_s_4' --num_envs 4 --learning_rate ::: 0.0002 0.0001 ::: --clip_range ::: 0.1 0.01 ::: --seed ::: {1..5} 
#parallel -j 5 python /home/cheryl16/projects/def-durandau/RL-Chemist/Train_reach_wandb.py --env_name 'UR10eReachFive-v1' --group 'experiment_5_mask_1' --num_envs 4 --learning_rate ::: 0.0002 0.0001 ::: --clip_range ::: 0.1 0.01 ::: --seed ::: {1..5} 
#parallel -j 5 python Train_reach_3.py --env_name 'UR10eReach3C-v0' --group 'experiment_reach3_6' --num_envs 3 --learning_rate 0.0001 --clip_range 0.1 --seed ::: {11..13} 

#parallel -j 5 python Train_reach_3.py --env_name 'UR10eSparse3C-v0' --group 'experiment_sparse3_1' --num_envs 4 --learning_rate 0.0002 --clip_range 0.1 --seed ::: {1..5} 
#python Train_reach_4.py --env_name 'UR10eSparse4C-v0' --group 'experiment_sparse_4' --num_envs 4 --learning_rate 0.0001 --clip_range 0.01 --seed $SLURM_ARRAY_TASK_ID

#parallel -j 5 python Train_reach_1h.py --env_name 'UR10eReach1H-v0' --group 'experiment_reach1_3' --num_envs 3 --learning_rate 0.0001 --clip_range 0.05 --seed ::: {1..5} 

parallel -j 5 python Train_reach_4.py --env_name 'UR10eReach4C-v1' --group 'ground_reach4_1' --num_envs 4 --learning_rate 0.0002 --clip_range 0.1 --seed ::: {1..5} 
#python Train_reach_4.py --env_name 'UR10eReach4C-v1' --group 'ground_reach4_1' --num_envs 4 --learning_rate 0.0002 --clip_range 0.1 --seed $SLURM_ARRAY_TASK_ID



#parallel -j 5 python Train_reach_3.py --env_name 'UR10eMask3C-v0' --group 'experiment_mask3_1' --num_envs 4 --learning_rate ::: 0.0001 0.0002 ::: --clip_range ::: 0.1 0.01 ::: --seed ::: {1..10} 
#parallel -j 5 python Train_reach_4.py --env_name 'UR10eMask4C-v0' --group 'experiment_mask4_1' --num_envs 4 --learning_rate ::: 0.0001 0.0002 ::: --clip_range ::: 0.1 0.01 ::: --seed ::: {1..5} 
