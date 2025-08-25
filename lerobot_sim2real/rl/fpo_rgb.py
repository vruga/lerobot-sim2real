from collections import defaultdict
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Local imports
from .network import FeedForwardNN

@dataclass
class PPOArgs:
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
    wandb_group: str = "PPO"
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

    # === FPO (Diffusion Policy) additions ===
    fpo_enable: bool = True                    # set False to fall back to PPO (if you keep code paths)
    fpo_num_steps: int = 8                     # diffusion Euler steps (reduced for stability)
    fpo_num_train_samples: int = 16            # CFM Monte-Carlo samples per transition (reduced for stability)
    fpo_fixed_noise_inference: bool = False    # reproducible eval for consistency
    fpo_logratio_clip: float = 0.3             # clamp for stability in exp(diff) (increased for more flexibility)
    positive_advantage: bool = False           # optional softplus on advantages




    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_kwargs: dict = field(default_factory=dict)
    """extra environment kwargs to pass to the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = None
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.00
    """coefficient of the entropy"""
    vf_coef: float = 1.2
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
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


class DiffusionPolicy(FeedForwardNN):
    """
    Extends FeedForwardNN for diffusion-based sampling with reproducible inference noise.
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 device: torch.device = None,
                 num_steps: int = 10,
                 fixed_noise_inference: bool = False):
        # Input dimension is state_dim + action_dim + 1 (for time)
        super().__init__(state_dim + action_dim + 1, action_dim)
        
        # select device and move model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # store whether to use fixed noise during inference
        self.fixed_noise_inference = fixed_noise_inference
        # pre-sample a single noise vector for inference
        self.init_noise = torch.randn(1, action_dim, device=self.device) * 0.01
        # num sampling step
        self.num_steps = num_steps
        self.state_dim = state_dim
        self.action_dim = action_dim

    def sample_action(self, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Run improved Euler diffusion to denoise initial noise into action.

        Args:
            state_norm: Tensor of shape (state_dim,), normalized state in [-1,1]

        Returns:
            Tensor of shape (action_dim,) representing action
        """
        if state_norm.ndim == 1:
            state_norm = state_norm.unsqueeze(0)
        
        num_steps = self.num_steps
        dt = 1.0 / num_steps
        
        # Initialize with improved noise scaling
        if self.fixed_noise_inference:
            x_t = self.init_noise.clone()
        else:
            # Use scaled noise for better initialization
            x_t = torch.randn(1, self.action_dim, device=self.device) * 0.5

        # Perform improved Euler integration with adaptive stepping
        for step in range(num_steps):
            t_val = (step + 0.5) / num_steps  # Use mid-point for better accuracy
            t_tensor = torch.full((1, 1), t_val, device=self.device)
            inp = torch.cat([state_norm.to(self.device), x_t, t_tensor], dim=1)
            
            with torch.no_grad():
                velocity = self(inp)
                # Apply gentle bounds to velocity for stability
                velocity = torch.clamp(velocity, -3.0, 3.0)
                
            # Euler step with momentum-like damping for stability
            x_t = x_t + dt * velocity * 0.9  # Slight damping factor

        return x_t[0]

    def sample_action_with_info(self, state_norm: torch.Tensor, num_train_samples: int = 100, include_inference_eps: bool = False):
        """
        Run Euler diffusion with tracking and return action + loss info.
        state_norm is (B, state_dim) or (state_dim,).. output action is (B, action_dim) or (action_dim,)

        Returns:
            pred_action: final denoised action
            x_t_path: all intermediate x_t steps [B, T+1, action_dim]  
            eps: sampled eps used for initial noise [B, action_dim]
            t: sampled time step [num_train_samples, 1]
            initial_cfm_loss: scalar [B, num_train_samples]
        """
        if state_norm.ndim == 1:
            state_norm = state_norm.unsqueeze(0)
        
        B = state_norm.size(0)
        dt = 1.0 / self.num_steps
        state_norm = state_norm.to(self.device)
        
        # Initialize noise for inference with improved scaling
        if self.fixed_noise_inference:
            eps = self.init_noise.expand(B, -1).contiguous()
        else:
            # Use more conservative noise scaling
            eps = torch.randn(B, self.action_dim, device=self.device) * 0.5
            
        x_t = eps.clone()
        x_t_path = [x_t.detach().clone()]

        # Perform improved Euler integration with better stepping
        for step in range(self.num_steps):
            t_val = (step + 0.5) / self.num_steps  # Mid-point integration
            t_tensor = torch.full((B, 1), t_val, device=self.device)
            inp = torch.cat([state_norm, x_t, t_tensor], dim=1)
            velocity = self(inp)
            # Apply velocity bounds for stability
            velocity = torch.clamp(velocity, -3.0, 3.0)
            x_t = x_t + dt * velocity * 0.9  # Slight damping
            x_t_path.append(x_t.detach().clone())

        x_t_path = torch.stack(x_t_path, dim=1)

        # IMPROVED flow matching training with better sampling strategy
        # Mix different noise scales and ensure proper coverage of the flow
        eps_sample = torch.randn(B, num_train_samples, self.action_dim, device=self.device)  # [B, N, A]
        # Scale noise with different magnitudes for better coverage
        noise_scales = torch.rand(B, num_train_samples, 1, device=self.device) * 0.8 + 0.2  # [0.2, 1.0]
        eps_sample = eps_sample * noise_scales
        
        # Better time sampling - avoid extremes and ensure good coverage
        t = torch.rand(B, num_train_samples, 1, device=self.device) * 0.9 + 0.05  # [0.05, 0.95]
        x1 = x_t.unsqueeze(1).expand(-1, num_train_samples, -1).detach()  # [B, N, A]
        state_tile = state_norm.unsqueeze(1).expand(-1, num_train_samples, -1)  # [B, N, state_dim]

        # Reshape for CFM loss computation
        BN = B * num_train_samples
        initial_cfm_loss = self.compute_cfm_loss(
            state_tile.reshape(BN, -1),
            x1.reshape(BN, -1), 
            eps_sample.reshape(BN, -1),
            t.reshape(BN, -1)
        ).view(B, num_train_samples)  # [B, N]

        # Return single sample if input was 1D
        if B == 1:
            return x_t[0], x_t_path, eps_sample, t, initial_cfm_loss.detach()
        else:
            return x_t, x_t_path, eps_sample, t, initial_cfm_loss.detach()
        
    def compute_cfm_loss(self, state_norm: torch.Tensor,
                         x1: torch.Tensor,
                         eps: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional flow matching loss with improved numerical stability.

        Args:
            state_norm: [B, state_dim] normalized input state
            x1: [B, action_dim] final denoised action
            eps: [B, action_dim] sampled noise
            t: [B, 1] time steps

        Returns:
            loss: [B] per-sample loss
        """
        B, D_a = eps.shape
        assert x1.shape == (B, D_a), f"x1 must be [B, D_a], got {x1.shape}"
        assert state_norm.shape[0] == B, f"state_norm must have batch size {B}, got {state_norm.shape[0]}"
        assert t.shape == (B, 1), f"t must be [B, 1], got {t.shape}"

        # Improved numerical stability
        t = torch.clamp(t, 0.001, 0.999)  # Prevent extreme t values
        
        # Linear interpolation between noise and target action
        x_t = (1 - t) * eps + t * x1  # [B, D_a]
        inp = torch.cat([state_norm, x_t, t], dim=1)  # [B, state_dim + D_a + 1]
        
        # Forward pass with gradient clipping
        velocity_pred = self(inp)  # [B, D_a]
        
        # Gentle velocity bounds to prevent explosion while allowing learning
        velocity_pred = torch.clamp(velocity_pred, -5.0, 5.0)

        # CORRECT CFM target velocity
        target_velocity = x1 - eps  # [B, D_a]
        
        # Normalize target to prevent exploding gradients
        target_norm = torch.norm(target_velocity, dim=1, keepdim=True)
        target_velocity = target_velocity / (target_norm + 1e-8) * torch.clamp(target_norm, max=5.0)
        
        # Robust MSE loss with gradient clipping
        loss = torch.nn.functional.huber_loss(velocity_pred, target_velocity, reduction='none', delta=1.0).mean(dim=1)
        
        # Light regularization to prevent overfitting
        velocity_reg = 0.0001 * torch.mean(velocity_pred.pow(2), dim=1)
        
        # Additional stability: detect and handle NaN/Inf
        loss = torch.clamp(loss + velocity_reg, 0.0, 10.0)
        loss = torch.where(torch.isnan(loss) | torch.isinf(loss), 
                          torch.zeros_like(loss), loss)
        
        return loss

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
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

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
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


#new class agent fpo
class Agent(nn.Module):
    def __init__(self, envs, sample_obs, args: PPOArgs = None):
        super().__init__()
        self.args = args
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features

        # Critic unchanged
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )

        # Diffusion actor
        action_dim = int(np.prod(envs.unwrapped.single_action_space.shape))
        self.action_low = torch.as_tensor(envs.unwrapped.single_action_space.low, dtype=torch.float32)
        self.action_high = torch.as_tensor(envs.unwrapped.single_action_space.high, dtype=torch.float32)

        n_steps = (args.fpo_num_steps if args is not None else 10)
        fixed = (args.fpo_fixed_noise_inference if args is not None else False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor_dp = DiffusionPolicy(
            state_dim=latent_size,
            action_dim=action_dim,
            device=device,
            num_steps=n_steps,
            fixed_noise_inference=fixed,
        )

    # ---- shared utils ----
    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def _clip_to_action_space(self, a: torch.Tensor) -> torch.Tensor:
        low = self.action_low.to(a.device)
        high = self.action_high.to(a.device)
        # Simple clipping to action bounds - let the policy learn the full range
        return torch.max(torch.min(a, high), low)

    # ---- PPO-compatible API (dummy logprob/entropy) ----
    @torch.no_grad()
    def get_action(self, x, deterministic=False):
        feats = self.feature_net(x)
        a, _, _, _, _ = self.actor_dp.sample_action_with_info(feats, num_train_samples=1)
        a = self._clip_to_action_space(a)
        return a  # determinism controlled by fpo_fixed_noise_inference

    @torch.no_grad()
    def get_action_and_value(self, x, action=None):
        feats = self.feature_net(x)
        a, _, _, _, _ = self.actor_dp.sample_action_with_info(feats, num_train_samples=1)
        a = self._clip_to_action_space(a)
        v = self.critic(feats)
        dummy_logprob = torch.zeros(a.size(0), device=a.device)
        dummy_entropy = torch.zeros_like(dummy_logprob)
        if action is None:
            action = a
        return action, dummy_logprob, dummy_entropy, v

    @torch.no_grad()
    def fpo_action_value_and_info(self, x):
        feats = self.feature_net(x)  # [B, L]
        a, _, eps, t_s, init_loss = self.actor_dp.sample_action_with_info(
            feats, num_train_samples=(self.args.fpo_num_train_samples if self.args else 16)
        )
        a = self._clip_to_action_space(a)
        v = self.critic(feats)
        return a, v, feats, eps, t_s, init_loss

    def compute_cfm_loss(self, state_norm, x1, eps, t):
        return self.actor_dp.compute_cfm_loss(state_norm, x1, eps, t)

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

def train(args: PPOArgs):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(
        obs_mode="rgb+segmentation", render_mode=args.render_mode, sim_backend="physx_cuda",
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    env_kwargs.update(args.env_kwargs)

    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
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
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=eval_envs.unwrapped.control_freq)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=eval_envs.unwrapped.control_freq, info_on_video=True)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    #new ===fpo storage ===
    # === FPO storages ===
    N = args.fpo_num_train_samples
    A = int(np.prod(envs.single_action_space.shape))
    fpo_eps = None     # [T, B, N, A]
    fpo_t = None       # [T, B, N, 1]
    fpo_init = None    # [T, B, N]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    #old agent
   # agent = Agent(envs, sample_obs=next_obs).to(device)
    #fpo agent
    agent = Agent(envs, sample_obs=next_obs, args=args).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

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
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
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
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done


            with torch.no_grad():
                action, value, feats, eps_s, t_s, init_loss = None, None, None, None, None, None
                action, value, feats, eps_s, t_s, init_loss = agent.fpo_action_value_and_info(next_obs)
                values[step] = value.view(-1)

            actions[step] = action
            logprobs[step] = torch.zeros(args.num_envs, device=device)


            # lazy-allocate FPO buffers once we know sizes
            if fpo_eps is None:
                fpo_eps  = torch.zeros(args.num_steps, args.num_envs, N, A, device=device)
                fpo_t    = torch.zeros(args.num_steps, args.num_envs, N, 1, device=device)
                fpo_init = torch.zeros(args.num_steps, args.num_envs, N, device=device)

            fpo_eps[step] = eps_s
            fpo_t[step] = t_s
            fpo_init[step] = init_loss

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # bootstrap value according to termination and truncation
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
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # FPO flatten
        b_eps   = fpo_eps.reshape(args.batch_size, N, A)     # [B, N, A]
        b_t     = fpo_t.reshape(args.batch_size, N, 1)       # [B, N, 1]
        b_init  = fpo_init.reshape(args.batch_size, N)       # [B, N]


        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        last_entropy_loss = 0.00
        last_old_approx_kl = 0.0
        last_approx_kl = 0.0

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb = b_inds[start:end]
                if len(mb) == 0: continue

                feats = agent.get_features(b_obs[mb])                     # [B, L]
                newvalue = agent.critic(feats).view(-1)                   # [B]
                x1 = b_actions[mb].view(feats.size(0), -1)                # [B, A]

                # Gather MC tensors for this minibatch
                eps_mb  = b_eps[mb]         # [B, N, A]
                t_mb    = b_t[mb]           # [B, N, 1]
                init_mb = b_init[mb]        # [B, N]

                B = feats.size(0)
                BN = B * N
                L = feats.size(1)
                A_ = x1.size(1)

                # Expand to BN then compute CFM loss
                feats_bn = feats.unsqueeze(1).expand(B, N, L).reshape(BN, L)
                x1_bn    = x1.unsqueeze(1).expand(B, N, A_).reshape(BN, A_)
                eps_bn   = eps_mb.reshape(BN, A_)
                t_bn     = t_mb.reshape(BN, 1)

                cfm_new = agent.compute_cfm_loss(feats_bn, x1_bn, eps_bn, t_bn).view(B, N)
                
                # Store raw loss for monitoring and debugging
                cfm_raw = cfm_new.clone().detach()
                
                # Compute importance ratio with improved stability
                diff = init_mb - cfm_new
                diff_raw = diff.clone().detach()  # Store raw for debugging
                
                # Apply gentle bounds to prevent extreme ratios while allowing learning
                diff = torch.clamp(diff, -3.0, 3.0)
                
                # Compute importance ratio with numerical stability
                log_ratio = diff.mean(dim=1)
                log_ratio = torch.clamp(log_ratio, -2.0, 2.0)  # Prevent extreme exp values
                rho = torch.exp(log_ratio)  # [B]
                
                # Apply PPO-style clipping
                clip_rho = torch.clamp(rho, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                
                # Advantages
                mb_advantages = b_advantages[mb]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                if args.positive_advantage:
                    mb_advantages = torch.nn.functional.softplus(mb_advantages)

                # Policy loss (rho replaces likelihood ratio) with gradient penalty
                pg_loss1 = -mb_advantages * rho
                pg_loss2 = -mb_advantages * clip_rho
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Improved regularization for stability without over-constraining
                # 1. Light KL penalty on importance ratios
                reg_loss = 0.001 * torch.mean(log_ratio.pow(2))
                
                # 2. Light action regularization
                action_reg = 0.0001 * torch.mean(x1.pow(2))
                
                # 3. Only add regularization if losses are reasonable
                if not torch.isnan(reg_loss) and not torch.isnan(action_reg):
                    total_reg = reg_loss + action_reg
                else:
                    total_reg = torch.tensor(0.0, device=device)
                
                pg_loss = pg_loss + total_reg
                
                # Store debug info for later printing (after all losses are computed)
                debug_info = None
                if start == 0 and epoch == 0:  # Only first minibatch, first epoch per iteration
                    debug_info = {
                        'iteration': iteration,
                        'init_mb': init_mb,
                        'cfm_raw': cfm_raw,
                        'diff_raw': diff_raw,
                        'log_ratio': log_ratio,
                        'rho': rho,
                        'x1': x1,
                        'reg_loss': reg_loss,
                        'action_reg': action_reg,
                        'pg_loss': pg_loss
                    }
                # wjhat thje fuck is thjis
                clipfracs += [((rho - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Value loss (unchanged)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                # No meaningful entropy/approx_kl in this path
                entropy_loss = torch.tensor(0.0, device=device)
                approx_kl = torch.tensor(0.0, device=device)
                old_approx_kl = torch.tensor(0.0, device=device)


                # after optimizer.step():
                last_entropy_loss = float(entropy_loss.item())
                last_old_approx_kl = float(old_approx_kl.item())
                last_approx_kl = float(approx_kl.item())

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # Check for NaN/Inf before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN/Inf loss detected, skipping update step")
                    continue
                    
                optimizer.zero_grad()
                loss.backward()
                
                # Check gradients before clipping
                total_grad_norm = 0.0
                for p in agent.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                # 🔍 PRINT DEBUG INFO NOW THAT ALL VALUES ARE COMPUTED
                if debug_info is not None:
                    print(f"\n🔍 ITERATION {debug_info['iteration']} - FPO DEBUG:")
                    print(f"CFM init loss: min={debug_info['init_mb'].min():.6f}, max={debug_info['init_mb'].max():.6f}, mean={debug_info['init_mb'].mean():.6f}")
                    print(f"CFM raw loss:  min={debug_info['cfm_raw'].min():.6f}, max={debug_info['cfm_raw'].max():.6f}, mean={debug_info['cfm_raw'].mean():.6f}")
                    print(f"CFM diff raw:  min={debug_info['diff_raw'].min():.6f}, max={debug_info['diff_raw'].max():.6f}, mean={debug_info['diff_raw'].mean():.6f}")
                    print(f"Log ratios:    min={debug_info['log_ratio'].min():.6f}, max={debug_info['log_ratio'].max():.6f}, mean={debug_info['log_ratio'].mean():.6f}")
                    print(f"Importance ratio: min={debug_info['rho'].min():.6f}, max={debug_info['rho'].max():.6f}, mean={debug_info['rho'].mean():.6f}")
                    print(f"Actions:       min={debug_info['x1'].min():.6f}, max={debug_info['x1'].max():.6f}, std={debug_info['x1'].std():.6f}")
                    print(f"Regularization: KL={debug_info['reg_loss']:.6f}, Action={debug_info['action_reg']:.6f}")
                    print(f"Policy loss: {debug_info['pg_loss']:.6f}, Value loss: {v_loss:.6f}, Total loss: {loss:.6f}")
                    print(f"Gradient norm: {total_grad_norm:.6f}")
                    print("=" * 60)
                
                # More aggressive gradient clipping for stability
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                
                # Skip update only if gradients are EXTREMELY large (after clipping)
                # Note: grad_norm here is the norm BEFORE clipping, but gradients are already clipped
                # Only skip if the raw gradient norm is astronomically high
                if total_grad_norm > 100.0:  # Much higher threshold - only skip truly extreme cases
                    print(f"WARNING: Extreme raw gradients detected ({total_grad_norm:.4f}), skipping step")
                    continue
                    
                optimizer.step()

            # Improved early stopping conditions
            # Stop if KL divergence becomes too high
            if reg_loss > args.target_kl:
                print(f"Early stopping at epoch {epoch} due to high KL divergence: {reg_loss:.4f}")
                break
            
            # Stop if CFM losses explode
            if cfm_raw.max() > 20.0 or torch.isnan(cfm_raw).any():
                print(f"Early stopping at epoch {epoch} due to unstable CFM loss: max={cfm_raw.max():.4f}")
                break
                
            # Stop if policy loss becomes NaN
            if torch.isnan(pg_loss) or torch.isinf(pg_loss):
                print(f"Early stopping at epoch {epoch} due to NaN/Inf policy loss")
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Standard metrics
        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        
        # FPO-specific metrics
        if 'cfm_raw' in locals():
            logger.add_scalar("fpo/cfm_loss_mean", cfm_raw.mean().item(), global_step)
            logger.add_scalar("fpo/cfm_loss_max", cfm_raw.max().item(), global_step)
            logger.add_scalar("fpo/cfm_loss_std", cfm_raw.std().item(), global_step)
        
        if 'log_ratio' in locals():
            logger.add_scalar("fpo/log_ratio_mean", log_ratio.mean().item(), global_step)
            logger.add_scalar("fpo/log_ratio_std", log_ratio.std().item(), global_step)
        
        if 'rho' in locals():
            logger.add_scalar("fpo/importance_ratio_mean", rho.mean().item(), global_step)
            logger.add_scalar("fpo/importance_ratio_max", rho.max().item(), global_step)
            
        if 'total_grad_norm' in locals():
            logger.add_scalar("training/grad_norm", total_grad_norm, global_step)
            
        # Action statistics
        logger.add_scalar("actions/mean_magnitude", torch.norm(b_actions, dim=1).mean().item(), global_step)
        logger.add_scalar("actions/std", b_actions.std().item(), global_step)
        
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("losses/entropy", last_entropy_loss, global_step)
        logger.add_scalar("losses/old_approx_kl", last_old_approx_kl, global_step)
        logger.add_scalar("losses/approx_kl", last_approx_kl, global_step)

        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()

