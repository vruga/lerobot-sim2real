# FPO++ Implementation

This repository includes an implementation of FPO++ (Flow Policy Optimization Plus Plus), an advanced reinforcement learning algorithm that builds upon the original FPO (Flow Policy Optimization) method.

## Key Features of FPO++

### 1. Per-Sample Ratio Clipping
- Unlike original FPO which averages CFM losses across all Monte Carlo samples before computing a single ratio per action, FPO++ computes separate ratios for each sample individually
- This allows each (τᵢ, ϵᵢ) pair to be clipped independently, providing finer-grained trust region control

### 2. Asymmetric Trust Region (ASPO)
- For positive advantages: Uses PPO-style clipping
- For negative advantages: Uses SPO objective to prevent aggressive CFM loss increases
- This prevents entropy collapse and enables better exploration properties

## Files Included

- `lerobot_sim2real/rl/fpo_plus_plus_rgb.py`: Main FPO++ implementation with visual observation support
- `lerobot_sim2real/rl/train_fpo_plus_plus.py`: Training script for FPO++
- `fpo_plus_plus_run.sh`: Run script for launching FPO++ training

## Usage

To train an FPO++ agent on the PickCube-v1 environment:

```bash
./fpo_plus_plus_run.sh
```

Or run directly:

```bash
python lerobot_sim2real/rl/train_fpo_plus_plus.py --env-id="PickCube-v1" --env-kwargs-json-path=env_config.json \
       --ppo.seed=999 \
       --ppo.num_envs=256 --ppo.num-steps=32 --ppo.update_epochs=2 --ppo.num_minibatches=8 \
       --ppo.total_timesteps=50_000_000 --ppo.gamma=0.95 --ppo.gae-lambda=0.95 \
       --ppo.learning-rate=1e-5 --ppo.reward-scale=1.0 \
       --ppo.num_eval_envs=16 --ppo.num-eval-steps=64 --ppo.no-partial-reset \
       --ppo.fpo-num-steps=6 --ppo.fpo-num-train-samples=8 --ppo.fpo-logratio-clip=0.5 \
       --ppo.clip-coef=0.15 --ppo.max-grad-norm=2.0 --ppo.target-kl=0.005 \
       --ppo.vf-coef=1.0 --ppo.ent-coef=0.005 \
       --ppo.fpo-per-sample-clipping=True --ppo.fpo-asymmetric-trust-region=True \
       --ppo.exp-name="fpo++-pickcube-v1-999" \
       --ppo.track --ppo.wandb_project_name "ManiSkill-FPO++"
```

## Key Parameters

- `fpo_per_sample_clipping`: Enable per-sample ratio clipping (True for FPO++)
- `fpo_asymmetric_trust_region`: Enable ASPO for asymmetric trust regions (True for FPO++)
- `fpo_num_steps`: Number of diffusion Euler steps
- `fpo_num_train_samples`: Number of CFM Monte-Carlo samples per transition
- `fpo_logratio_clip`: Clamp for stability in exp(diff)