import torch
import torch.nn as nn
# Note: Requires pip install torchdiffeq
# from torchdiffeq import odeint

class LiquidNode(nn.Module):
    """
    The fundamental cell of Unit 734.
    Instead of a static weight multiplication, this node maintains a continuous 
    internal state over time, evaluating an Ordinary Differential Equation (ODE).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # A simple linear layer representing the continuous dynamics (the 'flow')
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        # The node's memory/reservoir that decays over time without input
        self.register_buffer('state', torch.zeros(hidden_dim))

    def forward(self, t, x):
        """
        The differential equation logic defining dx/dt.
        """
        return self.dynamics(x)
