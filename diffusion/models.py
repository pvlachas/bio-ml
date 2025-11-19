import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps, device='cpu'):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out).to(device)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, n_steps, device='cpu'):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps, device=device)
        self.lin2 = ConditionalLinear(128, 128, n_steps, device=device)
        self.lin3 = nn.Linear(128, 4)

    def forward(self, x, t):
        x = F.softplus(self.lin1(x, t))
        x = F.softplus(self.lin2(x, t))
        return self.lin3(x)


class ConditionalModelMeanOnly(nn.Module):
    """
    Conditional model that predicts only the mean (not the variance).
    Used with fixed variance schedules.
    """
    def __init__(self, n_steps, device='cpu'):
        super(ConditionalModelMeanOnly, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps, device=device)
        self.lin2 = ConditionalLinear(128, 128, n_steps, device=device)
        self.lin3 = nn.Linear(128, 2)  # Only output mean (2D)

    def forward(self, x, t):
        x = F.softplus(self.lin1(x, t))
        x = F.softplus(self.lin2(x, t))
        return self.lin3(x)


class ConditionalModelNoisePredictor(nn.Module):
    """
    Conditional model that predicts the noise epsilon added during diffusion.

    This is the most common DDPM formulation. The model predicts ε such that:
        x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε

    The architecture is identical to ConditionalModelMeanOnly, but the
    interpretation is different.
    """
    def __init__(self, n_steps, device='cpu'):
        super(ConditionalModelNoisePredictor, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps, device=device)
        self.lin2 = ConditionalLinear(128, 128, n_steps, device=device)
        self.lin3 = nn.Linear(128, 2)  # Predict noise (same dim as input)

    def forward(self, x, t):
        x = F.softplus(self.lin1(x, t))
        x = F.softplus(self.lin2(x, t))
        return self.lin3(x)








