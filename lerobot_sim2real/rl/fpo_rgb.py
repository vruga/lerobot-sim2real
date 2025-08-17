# lerobot_sim2real/rl/fpo_rgb.py
"""
CleanRL-style **Flow Policy Optimization (FPO)** for visual RL in ManiSkill.

Core math (first principles):
  Path:  z(τ) = (1 - τ) ε + τ a,  with ε ~ N(0, I), τ ~ schedule in [τ_min, τ_max]
  True velocity along the path: ∂z/∂τ = a - ε

Conditional Flow Matching (CFM) loss trains a velocity field v_θ(z, τ, s)
to predict that true velocity:
  L_cfm(s, a; θ) = E_{τ,ε} [ || v_θ(z(τ), τ, s) - (a - ε) ||^2 ]

FPO surrogate ratio (ELBO proxy) replaces PPO's likelihood ratio:
  r_FPO(θ) = exp(  L_cfm(s,a; θ_old) - L_cfm(s,a; θ)  )

We plug r_FPO into the PPO-style clipped objective:
  max_θ E[ min( r_FPO * A, clip(r_FPO, 1±ε) * A ) ]

Key implementation details to avoid OOM and follow the paper:
- The **policy is a flow**: actions are sampled by integrating z' = v_θ(z, τ, s) from τ=0→1.
- During updates we snapshot θ_old once, and compute L_old & L_new **per minibatch**,
  reusing the **same (τ, ε)** pairs for both θ_old and θ.
- No Gaussian log-probs / entropy are used; stability comes from clipping r_FPO.
"""

from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


# ==============================
# Args
# ==============================
@dataclass
class FPOArgs:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "FPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments (mostly as in PPO backbone)
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_kwargs: dict = field(default_factory=dict)
    """extra environment kwargs to pass to the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes reconfigure eval env each reset to ensure randomization"""
    control_mode: Optional[str] = None
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """entropy coeff (unused in pure FPO; keep 0.0)"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """not used in pure FPO (no logπ); kept for interface compatibility"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # FPO / CFM additions
    cfm_noise_schedule: str = "linear"  # {"linear","cosine","sigmoid"}
    cfm_tau_min: float = 0.01
    cfm_tau_max: float = 0.99
    cfm_sample_taus: int = 4           # N_mc
    cfm_loss_coef: float = 0.0         # set to 0.0 for pure FPO (ELBO proxy already used)
    fpo_ratio_clip_coef: Optional[float] = None   # None => no pre-exp clamp (actual FPO)

    # Flow sampler (Euler steps for z' = vθ)
    flow_num_steps: int = 10

    # runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32
                        if v.dtype in (np.float32, np.float64)
                        else torch.uint8
                        if v.dtype == np.uint8
                        else torch.int16
                        if v.dtype == np.int16
                        else torch.int32
                        if v.dtype == np.int32
                        else v.dtype
                    )
                    self.data[k] = torch.zeros(
                        buffer_shape + v.shape, dtype=dtype, device=device
                    )

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


# Helper: accept DictArray or dict transparently
def _as_tensor_dict(obs_like):
    """Convert a DictArray to a plain dict of tensors (full slice)."""
    if isinstance(obs_like, DictArray):
        return obs_like[:]
    return obs_like


class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]

        # NatureCNN
        cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)  # keep identical to PPO
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class Agent(nn.Module):
    def __init__(self, envs, sample_obs, args: FPOArgs):
        super().__init__()
        self.args = args
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features

        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )

        # Flow velocity head v_θ(z, τ, s) — takes [feat, τ, z]
        act_dim = int(np.prod(envs.unwrapped.single_action_space.shape))
        self.act_dim = act_dim
        self.vel_head = nn.Sequential(
            layer_init(nn.Linear(latent_size + 1 + act_dim, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, act_dim), std=0.1),
        )

    # ------- Flow & CFM utilities -------

    def get_features(self, x: Dict[str, torch.Tensor]):
        x = _as_tensor_dict(x)
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def _sample_tau(self, b: int, device: torch.device):
        s = self.args.cfm_noise_schedule
        tmin, tmax = self.args.cfm_tau_min, self.args.cfm_tau_max
        u = torch.rand(b, device=device)
        if s == "linear":
            return tmin + (tmax - tmin) * u
        if s == "cosine":
            return tmin + 0.5 * (tmax - tmin) * (1 - torch.cos(u * np.pi))
        if s == "sigmoid":
            return tmin + (tmax - tmin) * torch.sigmoid(2 * (u - 0.5))
        return tmin + (tmax - tmin) * u

    @torch.no_grad()
    def _flow_sample(self, feat: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        Integrate z' = vθ(z, τ, s) from τ=0→1 with simple Euler steps.
        Start at ε (noise) if stochastic; at 0 if deterministic (ε=0).
        """
        B, A = feat.size(0), self.act_dim
        z = torch.zeros(B, A, device=feat.device)
        if not deterministic:
            z.normal_()  # ε ~ N(0, I)

        K = max(1, self.args.flow_num_steps)
        dt = 1.0 / K
        for k in range(K):
            tau = (k + 0.5) * dt  # midpoint Euler
            tau_tensor = torch.full((B, 1), tau, device=feat.device, dtype=feat.dtype)
            x = torch.cat([feat, tau_tensor, z], dim=-1)
            v = self.vel_head(x)
            z = z + dt * v
        return z  # action at τ=1

    def get_action(self, obs, deterministic=False):
        feat = self.get_features(obs)
        a = self._flow_sample(feat, deterministic=deterministic)
        return a

    def get_action_and_value(self, obs):
        # For rollouts: sample action (stochastic) and compute value
        with torch.no_grad():
            feat = self.get_features(obs)
            a = self._flow_sample(feat, deterministic=False)
            v = self.critic(feat).view(-1)
        return a, v

    def cfm_loss_from_feat_pairs(
        self,
        feat: torch.Tensor,             # [B, D]  precomputed features
        actions: torch.Tensor,          # [B, A]
        taus: torch.Tensor,             # [B, M]
        eps: torch.Tensor,              # [B, M, A]
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        """
        Per-sample Monte Carlo estimate of L_cfm(s, a; θ) given explicit (τ, ε) pairs.
        Uses precomputed features (no CNN inside) and chunks for OOM safety.
        Returns tensor [B] with per-item losses averaged over M.
        """
        device = actions.device
        B, A = actions.shape
        M = taus.shape[1]
        out = torch.empty(B, device=device, dtype=feat.dtype)

        for i0 in range(0, B, chunk_size):
            i1 = min(i0 + chunk_size, B)
            b = i1 - i0

            feat_b = feat[i0:i1]                        # [b, D]
            act_b  = actions[i0:i1]                     # [b, A]
            tau_b  = taus[i0:i1]                        # [b, M]
            eps_b  = eps[i0:i1]                         # [b, M, A]

            # z(τ) = (1 - τ) ε + τ a  (α_τ = τ, σ_τ = 1 - τ)
            z = (1 - tau_b.unsqueeze(-1)) * eps_b + tau_b.unsqueeze(-1) * act_b.unsqueeze(1)  # [b, M, A]
            target = act_b.unsqueeze(1) - eps_b                                             # [b, M, A]

            # expand for M
            feat_rep = feat_b.unsqueeze(1).expand(-1, M, -1).reshape(b * M, -1)             # [b*M, D]
            tau_rep  = tau_b.reshape(b * M, 1)                                              # [b*M, 1]
            z_rep    = z.reshape(b * M, A)                                                  # [b*M, A]
            x = torch.cat([feat_rep, tau_rep, z_rep], dim=-1)                               # [b*M, D+1+A]
            v_pred = self.vel_head(x).reshape(b, M, A)                                      # [b, M, A]

            per = F.mse_loss(v_pred, target, reduction="none").mean(dim=-1)                 # [b, M]
            out[i0:i1] = per.mean(dim=1)                                                    # [b]

        return out


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


def train(args: FPOArgs):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup (same as PPO backbone)
    env_kwargs = dict(
        obs_mode="rgb+segmentation", render_mode=args.render_mode, sim_backend="physx_cuda",
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    env_kwargs.update(args.env_kwargs)

    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=eval_envs.unwrapped.control_freq,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=eval_envs.unwrapped.control_freq,
            info_on_video=True,
        )
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running FPO training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.eval_partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["fpo"],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # Storage (PPO-style buffers + FPO pairs)
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # FPO: store (τ, ε) for each (t, env) with N_mc samples
    act_dim = int(np.prod(envs.single_action_space.shape))
    cfm_taus = torch.zeros((args.num_steps, args.num_envs, args.cfm_sample_taus), device=device)
    cfm_eps  = torch.zeros((args.num_steps, args.num_envs, args.cfm_sample_taus, act_dim), device=device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print("####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print("####")
    agent = Agent(envs, sample_obs=next_obs, args=args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True)
                    )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Anneal LR
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, value = agent.get_action_and_value(next_obs)
                values[step] = value
            actions[step] = action

            # env step
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            # store MC (τ, ε) pairs ONCE per (t, env)
            taus = agent._sample_tau(args.num_envs * args.cfm_sample_taus, device=device).view(
                args.num_envs, args.cfm_sample_taus
            )
            eps  = torch.randn(args.num_envs, args.cfm_sample_taus, act_dim, device=device)
            cfm_taus[step] = taus
            cfm_eps[step]  = eps

            if step % 10 == 0:
                print(f"Step {global_step}: Action norm: {action.norm().item():.2f}, Reward mean: {reward.mean().item():.3f}")

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(
                        infos["final_observation"]
                    ).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time

        # GAE / returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                if args.finite_horizon_gae:
                    if t == args.num_steps - 1:
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0
                        value_term_sum = 0.0
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                    )
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,))
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Flatten stored MC pairs
        B = args.batch_size
        M = args.cfm_sample_taus
        A = b_actions.shape[-1]
        b_taus = cfm_taus.reshape(B, M)
        b_eps  = cfm_eps.reshape(B, M, A)

        # Optimize — FPO: r = exp(L_old - L_new), with same (τ, ε)
        agent.train()
        b_inds = np.arange(B)
        clipfracs = []
        r_fpo_means = []
        update_time = time.perf_counter()

        # Snapshot θ_old ONCE per update
        agent_old = copy.deepcopy(agent).to(device).eval()
        for p in agent_old.parameters():
            p.requires_grad_(False)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, B, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                obs_mb = b_obs[mb_inds]
                act_mb = b_actions[mb_inds]
                taus_mb = b_taus[mb_inds]
                eps_mb  = b_eps[mb_inds]

                # features
                with torch.no_grad():
                    feat_old = agent_old.get_features(obs_mb)
                feat_new = agent.get_features(obs_mb)

                # L_old & L_new with SAME (τ, ε)
                with torch.no_grad():
                    L_old = agent_old.cfm_loss_from_feat_pairs(feat_old, act_mb, taus_mb, eps_mb)
                L_new = agent.cfm_loss_from_feat_pairs(feat_new, act_mb, taus_mb, eps_mb)

                # r_FPO = exp(L_old - L_new)  (optional pre-exp clamp if args.fpo_ratio_clip_coef not None)
                if args.fpo_ratio_clip_coef is None:
                    r_fpo = torch.exp(L_old - L_new)
                else:
                    r_fpo = torch.exp((L_old - L_new).clamp(-args.fpo_ratio_clip_coef, args.fpo_ratio_clip_coef))
                r_fpo_means.append(r_fpo.mean().item())

                # advantages
                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # PPO-style clipped surrogate but with r_fpo
                pg_loss1 = -mb_adv * r_fpo
                pg_loss2 = -mb_adv * torch.clamp(r_fpo, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (reuse feat_new to avoid extra CNN)
                newvalue = agent.critic(feat_new).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Optional CFM regularizer on current θ (pure FPO works with 0.0)
                cfm_reg = L_new.mean() * args.cfm_loss_coef

                loss = pg_loss + args.vf_coef * v_loss + cfm_reg

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Clipfrac for monitoring (how often |r-1| > εclip)
                with torch.no_grad():
                    clipfracs.append(((r_fpo - 1.0).abs() > args.clip_coef).float().mean().item())

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        logger.add_scalar("losses/cfm_loss", L_new.mean().item(), global_step)
        logger.add_scalar("fpo/r_fpo_mean", float(np.mean(r_fpo_means)) if r_fpo_means else 1.0, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar(
            "time/total_rollout+update_time",
            cumulative_times["rollout_time"] + cumulative_times["update_time"],
            global_step,
        )

    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_fpo_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None:
        logger.close()

