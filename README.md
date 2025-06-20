# GCRL_UR10e Setup Instructions
GCRL_UR10e is a research codebase for Goal-Conditioned Reinforcement Learning applied to the UR10e robotic arm. It implements methods from the paper [Versatile and Generalizable Manipulation via Goal-Conditioned Reinforcement Learning with Grounded Object Detection](https://openreview.net/forum?id=TgXIkK8WPQ&referrer=%5Bthe%20profile%20of%20Cheryl%20Wang%5D(%2Fprofile%3Fid%3D~Cheryl_Wang1)), presented at the CoRL 2024 Workshop. The repository provides training, evaluation, and pre-trained models for learning reaching tasks on the UR10e arm, supporting both standard and image-augmented training. It integrates GroundingDINO for visual goal representation and uses PPO for policy optimization.

A trained PPO policy and RL environment for the real UR10e robot are provided here: [Sim2Real_GCRL_UR10e](https://github.com/cherylwang20/Sim2Real_GCRL_UR10e). The policy can be used zero-shot or fine-tuned.

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
python training/Train_reach.py --env_name 'UR10eReach1C-v1' --group 'Reach_4C_dt20' --num_envs 4 --learning_rate 0.0003 --clip_range 0.1 --seed=$SLURM_ARRAY_TASK_ID --channel_num 4 --fs 20
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

To train with image augmentation, download the resized external images originally from [OpenX](https://robotics-transformer-x.github.io/) into `background` from https://mcgill-my.sharepoint.com/:u:/g/personal/huiyi_wang_mail_mcgill_ca/EZM8oZL_PPVIiOtrbl8Gy0sBLTBYWjd18TOdrS43WULVdA?e=ZBfhfY. 


Modify the training script:
```bash
python training/Train_reach.py --env_name "UR10eReach1C-v1" --merge True
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

python training/Train_reach.py --env_name 'UR10eReach1C-v1' --group 'Reach_4C_dt20' --num_envs 4 --learning_rate 0.0003 --clip_range 0.1 --seed=$SLURM_ARRAY_TASK_ID --channel_num 4 --fs 20
######Sim2Real
python training/Train_reach.py --env_name 'UR10eReach1C-v1' --group 'Reach_4C_dt20' --num_envs 4 --learning_rate 0.0003 --clip_range 0.1 --seed=$SLURM_ARRAY_TASK_ID --channel_num 4 --fs 20 --cont True
```

Submit with:
```bash
sbatch job.sh
```

# Sim2Real Transfer for UR10e Robotic Arm Using Mask-Based Goal-Conditioning

After the policy is trained, one could perform sim2real on an actual UR10e robot and a D435. This allows you to perform a mask-based GC PPO policy and environment to reach target objects placed on a table in front of a UR10e robot. 
cd into the folder sim2real to start the process. ```cd sim2real```

<img src="https://github.com/user-attachments/assets/40cceab5-dcca-40f2-a410-7acc832d7569" alt="UR10e" width="400"/>


## Features

- **UR10e Simulation Environment:** Customized simulation settings for the UR10e robotic arm.
- **External Policy Support:** Ability to download and integrate pre-trained policies for reaching tasks.
- **Action Type Flexibility:** Support for servoJ and moveJ position based control.
- **Real-world Adaptation:** Techniques and tools to adapt the simulation data for real-world application.

## Getting Started

These instructions will get you set up to run simulations with the UR10e robotic arm, including how to download external policies, load the environment, and apply different actions.

### Prerequisites

Ensure the following tools and libraries are installed:
- Python 3.8+
- CUDA 12.4

### Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/cherylwang20/Sim2Real_GCRL_UR10e.git
pip install -r requirements.txt
```

You would also need to use an external pre-trained object recognition model for object inference. We use GDINO here, which you can follow the link below to clone the repository and install:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```
### PyTorch 2.0 Fix

If you see an error from `value.type()` in `ms_deform_attn_cuda.cu`, replace it with `value.scalar_type()` to fix compatibility with PyTorch 2.0+ at `groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu`


### Downloading the External Policy

Use `wget` to download the pre-trained policy for the UR10e robotic arm:

```bash
mkdir -p policy
gdown 'https://drive.google.com/uc?id=1wKpIUVp2kXvf_Lq1VV7aKIoERLOS6QtW' -O policy/baseline.zip
```
### Getting Started

- The robot's initial joint configuration is:  
  `[4.7799, -2.0740, 2.6200, 3.0542, -1.5800, 1.4305e-05]` (in radians), with the gripper fully open.
- Place target objects **30–50 cm in front of the camera**, making sure they are **visible at the start**.
- The camera is mounted on the Robotiq gripper using a custom 3D-printed bracket.  
  It is essential that the **gripper is visible** in the camera view around 17 degrees downwards.

<img src="https://github.com/user-attachments/assets/d3fa1dee-6506-40b1-86d9-40dfb7742a22" alt="Camera Mounting" width="500"/>

- Set the correct IP address for your UR10e robot in:  
  [`GdinoReachGraspEnv_servoJ.py#L86`](https://github.com/cherylwang20/Sim2Real_GCRL_UR10e/blob/3f6d3c6f44f698b062e058aac546f5c7d1629576/src/reachGrasp_env/GdinoReachGraspEnv_servoJ.py#L86)

- Both `servoJ` and `moveJ` motion commands are supported.  
  **`servoJ` offers better performance for sim-to-real transfer.**
- We use a camera resolution of 848 * 480 for best inferenece results and later rescaled to 212 * 120 for policy training.
- Due to exceeding performance, we hardcorded a pick up after approaching close to the table and performing a pick up and drop up: https://github.com/cherylwang20/Sim2Real_GCRL_UR10e/blob/3f6d3c6f44f698b062e058aac546f5c7d1629576/src/reachGrasp_env/GdinoReachGraspEnv_servoJ.py#L326. Uncomment if you don't require this behavior. 

  
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
