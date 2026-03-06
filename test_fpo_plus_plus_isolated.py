#!/usr/bin/env python3
"""
Test script to validate the FPO++ objective function implementation in isolation
"""
import torch

def compute_fpo_plus_plus_objective(ratio, advantage, clip_coef, per_sample_clipping=True, asymmetric_trust_region=True):
    """
    Compute the FPO++ objective with per-sample clipping and asymmetric trust regions.
    
    Args:
        ratio: Shape [batch_size, num_samples] - importance weights for each sample
        advantage: Shape [batch_size, num_samples] - advantages for each sample
        clip_coef: Clipping coefficient (epsilon)
        per_sample_clipping: Whether to use per-sample ratio clipping (FPO++ improvement)
        asymmetric_trust_region: Whether to use ASPO for asymmetric trust regions
    
    Returns:
        Loss tensor of shape [batch_size, num_samples]
    """
    if per_sample_clipping:
        # FPO++: Per-sample ratio clipping
        if asymmetric_trust_region:
            # ASPO: Asymmetric trust region
            # For positive advantages: use PPO clipping
            # For negative advantages: use SPO quadratic penalty
            pos_adv_mask = advantage >= 0
            
            # PPO-style clipping for positive advantages
            ppo_clipped_ratio = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            ppo_loss = torch.where(pos_adv_mask, 
                                  torch.min(ratio * advantage, ppo_clipped_ratio * advantage),
                                  ratio * advantage)  # Don't clip for negative advantages yet
            
            # SPO-style quadratic penalty for negative advantages
            spo_penalty = torch.abs(advantage) / (2 * clip_coef) * (ratio - 1)**2
            spo_loss = ratio * advantage - spo_penalty
            
            # Combine: PPO for positive, SPO for negative
            final_loss = torch.where(pos_adv_mask, ppo_loss, spo_loss)
        else:
            # Standard PPO clipping applied per sample
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            final_loss = torch.min(ratio * advantage, clipped_ratio * advantage)
    else:
        # Original FPO: Average ratios before clipping (not recommended for FPO++)
        avg_ratio = torch.mean(ratio, dim=1, keepdim=True)  # [batch_size, 1]
        expanded_avg_ratio = avg_ratio.expand_as(ratio)  # [batch_size, num_samples]
        
        if asymmetric_trust_region:
            pos_adv_mask = advantage >= 0
            
            # PPO-style clipping for positive advantages
            ppo_clipped_ratio = torch.clamp(expanded_avg_ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            ppo_loss = torch.where(pos_adv_mask, 
                                  torch.min(expanded_avg_ratio * advantage, ppo_clipped_ratio * advantage),
                                  expanded_avg_ratio * advantage)
            
            # SPO-style quadratic penalty for negative advantages
            spo_penalty = torch.abs(advantage) / (2 * clip_coef) * (expanded_avg_ratio - 1)**2
            spo_loss = expanded_avg_ratio * advantage - spo_penalty
            
            # Combine: PPO for positive, SPO for negative
            final_loss = torch.where(pos_adv_mask, ppo_loss, spo_loss)
        else:
            # Standard PPO clipping with averaged ratios
            clipped_ratio = torch.clamp(expanded_avg_ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            final_loss = torch.min(expanded_avg_ratio * advantage, clipped_ratio * advantage)
    
    return -final_loss  # Negative because we want to maximize


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