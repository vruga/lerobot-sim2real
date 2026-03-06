#!/usr/bin/env python
"""
Test script to verify that the base_camera_settings fix works.
This creates a minimal version of the environment setup code from fpo_plus_plus_rgb.py
"""

import json
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PPOArgs:
    env_id: str = "PickCube-v1"
    env_kwargs: dict = field(default_factory=dict)

def test_env_creation():
    print("Testing environment creation with the fix...")
    
    # Load environment kwargs (similar to how it's done in the original script)
    env_kwargs = dict(
        obs_mode="rgb+segmentation", render_mode="all", sim_backend="physx_cuda",
    )
    
    # Load test config without base_camera_settings
    with open("env_config_test.json", "r") as f:
        loaded_kwargs = json.load(f)
    
    # Apply the fix logic from fpo_plus_plus_rgb.py
    base_camera_settings = loaded_kwargs.get('base_camera_settings', None)
    # Create a copy of env_kwargs without base_camera_settings to avoid passing it to the environment constructor
    filtered_env_kwargs = {k: v for k, v in loaded_kwargs.items() if k != 'base_camera_settings'}
    env_kwargs.update(filtered_env_kwargs)

    print(f"Environment kwargs: {env_kwargs}")
    print(f"Base camera settings extracted: {base_camera_settings is not None}")

    try:
        # Try to create the environment
        env = gym.make("PickCube-v1", num_envs=1, **env_kwargs)
        print("✓ Environment created successfully!")
        
        # Apply base_camera_settings after environment creation if the environment supports it
        if base_camera_settings is not None:
            if hasattr(env.unwrapped, 'base_camera_settings'):
                env.unwrapped.base_camera_settings = base_camera_settings
                print("✓ Applied base_camera_settings to environment")
            else:
                print("! Environment doesn't support base_camera_settings attribute")
        else:
            print("- No base_camera_settings to apply")
        
        env.close()
        print("✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        return False

if __name__ == "__main__":
    success = test_env_creation()
    if success:
        print("\n🎉 The fix appears to work correctly!")
    else:
        print("\n❌ The fix needs adjustment.")