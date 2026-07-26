import torch.nn as nn


class BaselineModel(nn.Module):
    """
        Baseline for HR estimation task
    """
    def __init__(self, input_dim, 
                 hidden_dim1, hidden_dim2, hidden_dim3, 
                 output_dim=1):
        super(BaselineModel, self).__init__()
        self.dropout = 0.2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(hidden_dim2, hidden_dim3),
            nn.LayerNorm(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(hidden_dim3, output_dim),
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)