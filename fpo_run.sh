#!/usr/bin/env bash
seed=2
python lerobot_sim2real/rl/train_fpo.py --env-id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json \
       --ppo.seed=${seed} \
       --ppo.num_envs=1024 --ppo.num-steps=16 --ppo.update_epochs=8 --ppo.num_minibatches=32 \
       --ppo.total_timesteps=50_000_000 --ppo.gamma=0.9 --ppo.gae-lambda=0.95 \
       --ppo.learning-rate=1e-5 --ppo.reward-scale=1.0 \
       --ppo.num_eval_envs=16 --ppo.num-eval-steps=64 --ppo.no-partial-reset \
       --ppo.fpo-num-steps=5 --ppo.fpo-num-train-samples=2 --ppo.fpo-logratio-clip=0.25 \
       --ppo.clip-coef=0.2 --ppo.max-grad-norm=1.0 --ppo.target-kl=0.02 \
             --ppo.fpo-fixed-noise-inference \
       --ppo.exp-name="fpo-SO100GraspCube-v1-rgb-stable-${seed}" \
       --ppo.track --ppo.wandb_project_name "SO100-ManiSkill-FPO"

