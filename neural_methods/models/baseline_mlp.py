import torch.nn as nn


class BaselineModel(nn.Module):
    """MLP baseline for PPG → HR estimation."""

    def __init__(self, config, input_dim=None):
        super().__init__()
        self.input_dim = input_dim if input_dim is not None else getattr(config, "input_dim", None)  # [DUAL-RATE-DESIGN]
        self.dropout = 0.2

        if self.input_dim is None:
            raise ValueError("BaselineModel requires an input_dim; it is derived in BaseTrainer via utils/preprocessing_config.py.")

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_dim1),
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