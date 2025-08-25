import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights with orthogonal initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeedForwardNN(nn.Module):
    """
    Enhanced feedforward neural network class for diffusion policies.
    CRITICAL FIX: Deeper architecture with better capacity for flow matching.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Improved time embedding with better conditioning
        self.time_embed = nn.Sequential(
            layer_init(nn.Linear(1, 128), std=0.3),
            nn.SiLU(),
            layer_init(nn.Linear(128, 128), std=0.3),
            nn.SiLU(),
            layer_init(nn.Linear(128, 64), std=0.3),
        )
        
        # Main state+action processing (excluding time)
        main_in_dim = in_dim - 1  # Subtract 1 for time dimension
        
        # Improved architecture with better flow capacity
        self.main_layers = nn.ModuleList([
            layer_init(nn.Linear(main_in_dim + 64, 512), std=0.3),  # +64 for time embedding
            layer_init(nn.Linear(512, 512), std=0.3),
            layer_init(nn.Linear(512, 512), std=0.3),
            layer_init(nn.Linear(512, 256), std=0.3),
            layer_init(nn.Linear(256, 128), std=0.3),
            layer_init(nn.Linear(128, out_dim), std=0.01),  # Small final weights for stability
        ])
        
        self.activations = nn.ModuleList([
            nn.SiLU(),
            nn.SiLU(), 
            nn.SiLU(),
            nn.SiLU(),
            nn.SiLU(),
            None  # No activation on output
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.05),  # Reduced dropout for better flow learning
            nn.Dropout(0.05),
            nn.Dropout(0.05),
            nn.Dropout(0.02),
            nn.Dropout(0.02),
            None
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input: [state, action, time]
        main_input = x[:, :-1]  # Everything except last dimension (time)
        time_input = x[:, -1:] # Last dimension (time)
        
        # Process time embedding
        t_emb = self.time_embed(time_input)  # [B, 64]
        
        # Combine main input with time embedding
        h = torch.cat([main_input, t_emb], dim=1)
        
        # Forward through layers with residual connections
        for i, (layer, activation, dropout) in enumerate(zip(self.main_layers, self.activations, self.dropouts)):
            h_new = layer(h)
            
            # Add residual connection where dimensions match
            if i > 0 and h.size(-1) == h_new.size(-1):
                h_new = h_new + h  # Residual connection
                
            if activation is not None:
                h_new = activation(h_new)
            if dropout is not None:
                h_new = dropout(h_new)
                
            h = h_new
            
        return h