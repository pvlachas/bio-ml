import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalLinear(nn.Module):
    """Linear layer with time embedding as FiLM like scale."""

    def __init__(self, d_in, d_out, T):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.emb = nn.Embedding(T, d_out)
        nn.init.uniform_(self.emb.weight, -0.1, 0.1)

    def forward(self, x, t):
        # project x and scale by time embedding
        y = self.lin(x)
        gamma = self.emb(t).view(x.size(0), -1)
        return gamma * y

class MeanLogVarNet(nn.Module):
    """Simple MLP to predict mean and log variance of p(x_{t-1} | x_t)."""
    def __init__(self, T, dim, hidden_dim=128):
        super().__init__()
        self.l1 = ConditionalLinear(dim, hidden_dim, T)
        self.l2 = ConditionalLinear(hidden_dim, hidden_dim, T)
        self.out = nn.Linear(hidden_dim, 2*dim) # mean and log var


    def forward(self, x, t):
        # two hidden layers with softplus activations
        x = F.softplus(self.l1(x, t))
        # second layer
        x = F.softplus(self.l2(x, t))
        # output layer
        out = self.out(x)
        # split mean and log var
        mean, log_var = torch.chunk(out, 2, dim=-1)
        return mean, log_var