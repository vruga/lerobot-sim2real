#!/bin/bash
seed=3
python lerobot_sim2real/scripts/train_fpo_rgb.py --env-id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json \
  --fpo.seed=${seed} \
  --fpo.num_envs=1024 --fpo.num-steps=16 --fpo.update_epochs=8 --fpo.num_minibatches=32 \
  --fpo.total_timesteps=5_000_000 --fpo.gamma=0.9 \
  --fpo.num_eval_envs=16 --fpo.num-eval-steps=64 --fpo.no-partial-reset \
  --fpo.cfm_loss_coef=1.0 \
  --fpo.cfm_sample_taus=4 --fpo.fpo_ratio_clip_coef=0.2 \
  --fpo.exp-name="fpo-SO100GraspCube-v1-rgb-${seed}" \
  --fpo.track --fpo.wandb_project_name "SO100-ManiSkill" \
