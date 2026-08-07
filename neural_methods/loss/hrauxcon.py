import torch
from torch import nn

class HRAuxConLoss(nn.Module):
    def __init__(self, lambda_t=0.2, lambda_f=0.2, lambda_c=1.0):
        super().__init__()

        # Fixed lambda
        self.lambda_t = lambda_t
        self.lambda_f = lambda_f
        self.lambda_c = lambda_c


        self.mse = nn.MSELoss()

    def forward(self, hr_pred, hr_time, hr_freq, hr_gt):
        # Main loss
        loss_main = self.mse(hr_pred, hr_gt)

        # Auxiliary loss
        loss_aux = (
            self.lambda_t * self.mse(hr_time, hr_gt)
            + self.lambda_f * self.mse(hr_freq, hr_gt)
        )

        # Consistency loss
        loss_con = self.lambda_c * self.mse(hr_time, hr_freq)

        total_loss = loss_main + loss_aux + loss_con

        return total_loss, {
            "main": loss_main,
            "aux": loss_aux,
            "con": loss_con,
        }