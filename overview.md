# 🧠 Versatile and Generalizable Manipulation via Goal-Conditioned Reinforcement Learning with Grounded Object Detection

This repository implements the method from the paper:  
**"Versatile and Generalizable Manipulation via Goal-Conditioned Reinforcement Learning with Grounded Object Detection"**  
Huiyi Wang, Fahim Shahriar, Seyed Alireza Azimi, Gautham Vasan, A. Rupam Mahmood, Colin Bellinger  
Accepted at the CoRL 2024 Workshop on Minimalist Robot Learning (MRM-D).  
📄 [Read the paper](https://openreview.net/pdf?id=TgXIkK8WPQ)

---

## 🚀 Overview

This project explores how **goal-conditioned reinforcement learning (GCRL)** can be enhanced using **pre-trained grounded object detectors**, enabling a single manipulation policy to generalize to a wide variety of objects specified via natural language.

---

## 🎯 Key Idea

Traditional GCRL struggles to generalize across different target objects. This work integrates a **pre-trained object grounding model** (GroundingDINO + SAM) to convert a **text goal** (e.g., “apple on the right”) into a **binary goal mask** representing the object’s current position in the visual field.

The policy is trained to condition on:
- RGB image
- Proprioceptive state
- Binary goal mask (updated each timestep)

---

## 🧩 Method Summary

- **Text Prompt → Object Detector → Binary Mask**  
  Uses a grounded vision-language model to localize target objects.
  
- **Goal Conditioning Variants**:
  - One-hot vector (baseline)
  - Goal object image crop
  - **Binary goal mask** (proposed)

- **Reinforcement Learning Algorithm**:  
  PPO (Proximal Policy Optimization) using RGB, proprioception, and mask as input.

---

## 📊 Results

| Goal Type           | Seen Objects (In-Dist) | Unseen Objects (OOD) |
|---------------------|------------------------|-----------------------|
| One-hot Vector      | 13%                    | 20%                   |
| Goal Object Image   | 62%                    | 28%                   |
| **GT Binary Mask**  | **89%**                | **90%**               |

- Binary masks from object detectors enable strong **zero-shot generalization** to unseen objects.
- Policies trained on **GT masks** transfer well to **DINO-generated masks** (∼90% success on seen objects), but real-time inference masks remain noisy in cluttered scenes.

---

## 📦 Key Features

- 🔍 **Pre-trained object grounding** with GroundingDINO + SAM.
- 🎯 **Mask-based goal representation** enables generalization.
- 🤖 RL agent that can grasp objects based on **natural language prompts**.

---

## 📁 Code and Usage

Coming soon.

---

## 📝 Citation

```bibtex
@inproceedings{
    wang2024goalconditioned,
    title={Versatile and Generalizable Manipulation via Goal-Conditioned Reinforcement Learning with Grounded Object Detection},
    author={Huiyi Wang and Fahim Shahriar and Seyed Alireza Azimi and Gautham Vasan and A. Rupam Mahmood and Colin Bellinger},
    booktitle={CoRL 2024 Workshop on Minimalist Robot Learning (MRM-D)},
    year={2024},
    url={https://openreview.net/forum?id=TgXIkK8WPQ}
}
