#!/bin/bash
# FPO++ — PickCube-v1, rgb+segmentation, paper-aligned hyperparameters
# Run 4 fixes: clip_coef=0.2 (was 0.05, was causing 55% clipfrac), target_kl=0.05

PYTORCH_ALLOC_CONF=expandable_segments:True \
.venv/bin/python lerobot_sim2real/rl/train_fpo_plus_plus.py \
  --env-id="PickCube-v1" \
  --ppo.seed=42 \
  --ppo.num_envs=256 \
  --ppo.num-steps=50 \
  --ppo.update_epochs=4 \
  --ppo.num_minibatches=50 \
  --ppo.total_timesteps=100_000_000 \
  --ppo.gamma=0.99 \
  --ppo.gae-lambda=0.95 \
  --ppo.learning-rate=3e-4 \
  --ppo.critic-learning-rate=1e-3 \
  --ppo.anneal-lr \
  --ppo.reward-scale=1.0 \
  --ppo.num_eval_envs=16 \
  --ppo.num-eval-steps=50 \
  --ppo.no-partial-reset \
  --ppo.fpo-num-steps=10 \
  --ppo.fpo-num-train-samples=16 \
  --ppo.fpo-logratio-clip=2.0 \
  --ppo.clip-coef=0.2 \
  --ppo.max-grad-norm=0.5 \
  --ppo.target-kl=0.05 \
  --ppo.vf-coef=0.5 \
  --ppo.ent-coef=0.0 \
  --ppo.fpo-per-sample-clipping \
  --ppo.fpo-asymmetric-trust-region \
  --ppo.exp-name="fpo++-pickcube-v4" \
  --ppo.track \
  --ppo.wandb_project_name "ManiSkill-FPO++"
