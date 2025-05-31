# GCRL_UR10e Setup Instructions

## Clone the Repository

```bash
git clone https://github.com/cherylwang20/GCRL_UR10e.git
cd GCRL_UR10e
git submodule update --init --recursive
```

## Set Up the Virtual Environment

Use **Python 3.9** (later versions may cause issues with loading the baseline):

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Compute Canada, use:
```bash
pip install -r requirements_CC.txt
```

## Load Submodules

### UR10e Gym Environment (mj_envs)

```bash
cd mj_envs
pip install -e .
```

### GroundingDINO

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

**Note on PyTorch 2.0 Compatibility:**  
If you encounter an error with `value.type()` in `ms_deform_attn_cuda.cu`, replace it with `value.scalar_type()` in:
```
groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
```

## Download the Pre-trained Policy

```bash
mkdir -p policy
gdown 'https://drive.google.com/uc?id=1wKpIUVp2kXvf_Lq1VV7aKIoERLOS6QtW' -O policy/baseline.zip
```

## Training a New Policy

To train a new policy, run:
```bash
--env_name 'UR10eReach1C-v1' --group 'Reach_4C_dt20' --num_envs 4 --learning_rate 0.0003 --clip_range 0.1 --seed=$SLURM_ARRAY_TASK_ID --channel_num 4 --fs 20
```
Training Script Arguments

```--env_name 'UR10eReach1C-v1'``` : Specifies the UR10e environment for training.

```--group 'Reach_4C_dt20'``` : Name of the experiment group for logging.

```--num_envs 4``` : Number of parallel environments.

```--learning_rate 0.0003``` : Learning rate for PPO.

```--clip_range 0.1``` : PPO clip range for stable policy updates.

```--seed 0``` : Random seed, often set via SLURM for batch runs.

```--channel_num 4``` : Number of input image channels.

```--fs 20``` : Frame skip (simulation step interval).


## Training with Image Augmentation

To train with image augmentation, download the resized external images from [OpenX](https://robotics-transformer-x.github.io/). Place them in the appropriate directory (e.g., `data/images/`).

Modify the training script:
```bash
python training/Train_reach_int.py --env_name "UR10eReach1C-v1" --merge True
```

## Evaluate an Existing Policy

```bash
python training/Eval_reach_int.py --env_name "UR10eReach1C-v1" --model_num "baseline"
```

## Training on Compute Canada

Modify `job.sh` to set the desired environment and training script, fill in your own account information and path.

```bash
#!/bin/bash 
#SBATCH --account=your-account
#SBATCH --job-name=your-job-name
#SBATCH --cpus-per-task=32
#SBATCH --time=0-50:50
#SBATCH --array=5-9
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=your-email
#SBATCH --mail-type=ALL

export PYTHONPATH="$PYTHONPATH:/yourpath/"

cd /yourpath/

module load StdEnv/2023
module load gcc opencv/4.9.0 cuda/12.2 python/3.10 mpi4py mujoco/3.1.6

source /your_env/

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

wandb offline

######Sim2Real
#python training/Train_reach.py --env_name 'UR10eReach1C-v1' --group 'Reach_4C_dt20' --num_envs 4 --learning_rate 0.0003 --clip_range 0.1 --seed=$SLURM_ARRAY_TASK_ID --channel_num 4 --fs 20

python training/Train_reach.py --env_name 'UR10ePickPlace-v0' --group 'Pick_dis_cher' --num_envs 4 --learning_rate 0.0003 --clip_range 0.1 --seed=$SLURM_ARRAY_TASK_ID --channel_num 4 --fs 20
```

Submit with:
```bash
sbatch job.sh
```

# Citation
If you use this repository or any of its code in your work, please cite:

```bibtex
@inproceedings{
    wang2024versatile,
    title={Versatile and Generalizable Manipulation via Goal-Conditioned Reinforcement Learning with Grounded Object Detection},
    author={HUIYI WANG and Fahim Shahriar and Seyed Alireza Azimi and Gautham Vasan and A. Rupam Mahmood and Colin Bellinger},
    booktitle={CoRL 2024 Workshop on Mastering Robot Manipulation in a World of Abundant Data},
    year={2024},
    url={https://openreview.net/forum?id=TgXIkK8WPQ}
}
```
