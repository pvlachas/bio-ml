import torch.nn as nn

# Define a simple neural network
class ScoreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ScoreNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)
