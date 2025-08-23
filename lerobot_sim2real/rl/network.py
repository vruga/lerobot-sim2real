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
    Base feedforward neural network class for diffusion policies.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Build a simple feedforward network
        self.network = nn.Sequential(
            layer_init(nn.Linear(in_dim, 256), std=0.5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            layer_init(nn.Linear(256, 128), std=0.5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            layer_init(nn.Linear(128, out_dim), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)