### Testing Policies for Sim2Real

This repository provides two testing scripts for deploying trained policies on the UR10e robot using GroundingDINO inference:

#### `test_policy_GDINO_servoJ.py`
- Uses `servoJ` control (real-time velocity-based joint commands)
- Recommended for sim-to-real due to smoother and more stable execution
- Run with:
  ```bash
  python test_policy_GDINO_servoJ.py
  ```
#### `test_policy_GDINO.py`
- Uses `moveJ` control (real-time position-based joint commands)
- One-to-one transfer of policy from simulation results in jerky motion and long lags between actions.
- Run with:
  ```bash
  python test_policy_servoJ.py
  ```
