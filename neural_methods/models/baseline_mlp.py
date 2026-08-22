import torch.nn as nn


class BaselineModel(nn.Module):
    """MLP baseline for PPG → HR estimation."""

    def __init__(self, config):
        super().__init__()
        self.dropout = 0.2

        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim1),
            nn.LayerNorm(config.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(config.hidden_dim1, config.hidden_dim2),
            nn.LayerNorm(config.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(config.hidden_dim2, config.hidden_dim3),
            nn.LayerNorm(config.hidden_dim3),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(config.hidden_dim3, getattr(config, "output_dim", 1)),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)