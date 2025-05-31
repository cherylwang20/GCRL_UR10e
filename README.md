## Clone the repository

```
git clone https://github.com/cherylwang20/GCRL_UR10e.git
git submodule update --init --recursive
```

## Creating a new venv
```
python3 -m venv .venv
source .venv/bin/activate
```

# Loading the submodules

The UR10e gym environment can be used via the mj_envs package. 

`cd mj_envs` and `pip install -e .`

We use GDINO here, which you can follow the link below to clone the repository and install:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```
### PyTorch 2.0 Fix

If you see an error from `value.type()` in `ms_deform_attn_cuda.cu`, replace it with `value.scalar_type()` to fix compatibility with PyTorch 2.0+ at `groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu`


## Downloading the External Policy

Use `wget` to download the pre-trained policy for the UR10e robotic arm:

```bash
mkdir -p policy
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1wKpIUVp2kXvf_Lq1VV7aKIoERLOS6QtW" -O policy/baseline.zip

```

## To train with image augmentation, please download the resized external image from (OpenX)[https://robotics-transformer-x.github.io/] here:




## Training on Alliance Canada



# UR10e Env
Reach Environment: `env = gym.make(f'mj_envs.robohive.envs:{"UR10eReachFixed-v0"}')`
