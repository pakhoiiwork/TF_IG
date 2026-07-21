import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(BaselineModel, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)