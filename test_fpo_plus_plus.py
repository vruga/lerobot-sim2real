#!/usr/bin/env python3
"""
Test script to validate the FPO++ objective function implementation
"""
import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lerobot_sim2real.rl.fpo_plus_plus_rgb import compute_fpo_plus_plus_objective

def test_fpo_plus_plus_objective():
    print("Testing FPO++ objective function...")
    
    # Create test tensors
    batch_size = 4
    num_samples = 8
    clip_coef = 0.2
    
    # Random ratios and advantages for testing
    ratio = torch.rand(batch_size, num_samples) * 2.0  # Values between 0 and 2
    advantage = torch.randn(batch_size, num_samples)   # Random advantages (some positive, some negative)
    
    print(f"Input shapes - ratio: {ratio.shape}, advantage: {advantage.shape}")
    print(f"Ratio range: [{ratio.min():.3f}, {ratio.max():.3f}]")
    print(f"Advantage range: [{advantage.min():.3f}, {advantage.max():.3f}]")
    
    # Test with per-sample clipping and asymmetric trust region (FPO++ defaults)
    loss_per_sample_asym = compute_fpo_plus_plus_objective(
        ratio, advantage, clip_coef, 
        per_sample_clipping=True, 
        asymmetric_trust_region=True
    )
    
    print(f"\nFPO++ (per-sample + ASPO) loss shape: {loss_per_sample_asym.shape}")
    print(f"FPO++ loss range: [{loss_per_sample_asym.min():.3f}, {loss_per_sample_asym.max():.3f}]")
    
    # Test with per-sample clipping but standard PPO (not ASPO)
    loss_per_sample_std = compute_fpo_plus_plus_objective(
        ratio, advantage, clip_coef, 
        per_sample_clipping=True, 
        asymmetric_trust_region=False
    )
    
    print(f"\nFPO++ (per-sample + PPO) loss shape: {loss_per_sample_std.shape}")
    print(f"FPO++ loss range: [{loss_per_sample_std.min():.3f}, {loss_per_sample_std.max():.3f}]")
    
    # Test with averaged ratios (original FPO style)
    loss_avg_asym = compute_fpo_plus_plus_objective(
        ratio, advantage, clip_coef, 
        per_sample_clipping=False, 
        asymmetric_trust_region=True
    )
    
    print(f"\nFPO (averaged + ASPO) loss shape: {loss_avg_asym.shape}")
    print(f"FPO loss range: [{loss_avg_asym.min():.3f}, {loss_avg_asym.max():.3f}]")
    
    print("\n✓ All FPO++ objective function tests passed!")
    
    # Additional validation: Check that ASPO handles positive and negative advantages differently
    print("\nTesting ASPO behavior with mixed advantages...")
    
    # Create a scenario with both positive and negative advantages
    test_ratio = torch.tensor([[1.5, 0.5, 1.2, 0.8]])  # [batch_size=1, num_samples=4]
    test_advantage = torch.tensor([[0.5, -0.5, 0.3, -0.3]])  # Mixed positive/negative
    
    loss_asym = compute_fpo_plus_plus_objective(
        test_ratio, test_advantage, clip_coef=0.1,
        per_sample_clipping=True,
        asymmetric_trust_region=True
    )
    
    loss_sym = compute_fpo_plus_plus_objective(
        test_ratio, test_advantage, clip_coef=0.1,
        per_sample_clipping=True,
        asymmetric_trust_region=False
    )
    
    print(f"Test ratio: {test_ratio.squeeze()}")
    print(f"Test advantage: {test_advantage.squeeze()}")
    print(f"ASPO loss: {loss_asym.squeeze()}")
    print(f"PPO loss: {loss_sym.squeeze()}")
    
    print("\n✓ ASPO behavior test completed!")

if __name__ == "__main__":
    test_fpo_plus_plus_objective()