"""Simple script to train a RGB FPO (Flow Matching Policy Optimization) policy in simulation


"""

from dataclasses import dataclass, field
import json
from typing import Optional
import tyro

from lerobot_sim2real.rl.fpo_rgb import PPOArgs as FPOArgs, train

@dataclass
class Args:
    env_id: str
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    fpo: FPOArgs = field(default_factory=FPOArgs)
    """FPO training arguments"""

def main(args: Args):
    args.fpo.env_id = args.env_id
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs = json.load(f)
        args.fpo.env_kwargs = env_kwargs
    else:
        print("No env kwargs json path provided, using default env kwargs with default settings")
    train(args=args.fpo)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
