#!/usr/bin/env bash
seed=3
python lerobot_sim2real/rl/train_fpo.py --env-id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json \
       --ppo.seed=${seed} \
       --ppo.num_envs=512 --ppo.num-steps=32 --ppo.update_epochs=4 --ppo.num_minibatches=16 \
       --ppo.total_timesteps=50_000_000 --ppo.gamma=0.95 --ppo.gae-lambda=0.95 \
       --ppo.learning-rate=1e-4 --ppo.reward-scale=1.0 \
       --ppo.num_eval_envs=8 --ppo.num-eval-steps=64 --ppo.no-partial-reset \
       --ppo.fpo-num-steps=5 --ppo.fpo-num-train-samples=8 --ppo.fpo-logratio-clip=2.0 \
       --ppo.fpo-fixed-noise-inference \
       --ppo.clip-coef=0.1 --ppo.max-grad-norm=0.3 --ppo.target-kl=0.05 \
       --ppo.exp-name="fpo-SO100GraspCube-v1-rgb-stable-${seed}" \
       --ppo.track --ppo.wandb_project_name "SO100-ManiSkill-FPO"
