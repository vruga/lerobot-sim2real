#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-3}"
ENV_ID="${2:-SO100GraspCube-v1}"
ENV_KWARGS_JSON="${3:-env_config.json}"

RUN_NAME="fpo-${ENV_ID}-rgb-${SEED}"

python lerobot_sim2real/scripts/train_fpo_rgb.py \
  --env-id "${ENV_ID}" \
  --env-kwargs-json-path "${ENV_KWARGS_JSON}" \
  --fpo.seed "${SEED}" \
  --fpo.exp-name "${RUN_NAME}" \
  --fpo.track \
  --fpo.wandb-project-name "SO100-ManiSkill" \
  --fpo.num-envs 1024 \
  --fpo.num-steps 32 \
  --fpo.update-epochs 6 \
  --fpo.num-minibatches 64 \
  --fpo.total-timesteps 10000000 \
  --fpo.gamma 0.99 \
  --fpo.num-eval-envs 8 \
  --fpo.num-eval-steps 100 \
  --fpo.eval-freq 25 \
  --fpo.no-partial-reset \
  --fpo.learning-rate 3e-4 \
  --fpo.vf-coef 0.7 \
  --fpo.ent-coef 0.0 \
  --fpo.clip-coef 0.1 \
  --fpo.fpo-num-steps 8 \
  --fpo.fpo-num-train-samples 32 \
  --fpo.fpo-logratio-clip 0.7 \
  --fpo.render-mode all \
  --fpo.no-capture-video

