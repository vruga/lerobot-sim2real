#!/usr/bin/env bash
seed=999


python lerobot_sim2real/rl/train_fpo.py --env-id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json \
       --ppo.seed=${seed} \
       --ppo.num_envs=256 --ppo.num-steps=32 --ppo.update_epochs=2 --ppo.num_minibatches=8 \
       --ppo.total_timesteps=50_000_000 --ppo.gamma=0.95 --ppo.gae-lambda=0.95 \
       --ppo.learning-rate=1e-5 --ppo.reward-scale=1.0 \
       --ppo.num_eval_envs=16 --ppo.num-eval-steps=64 --ppo.no-partial-reset \
       --ppo.fpo-num-steps=6 --ppo.fpo-num-train-samples=8 --ppo.fpo-logratio-clip=0.5 \
       --ppo.clip-coef=0.15 --ppo.max-grad-norm=2.0 --ppo.target-kl=0.005 \
       --ppo.vf-coef=1.0 --ppo.ent-coef=0.005 \
       --ppo.exp-name="fpo-rescue-SO100GraspCube-v1-${seed}" \
       --ppo.track --ppo.wandb_project_name "SO100-ManiSkill-FPO-Rescue"







