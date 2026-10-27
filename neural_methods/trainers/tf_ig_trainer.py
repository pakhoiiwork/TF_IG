from neural_methods.models.tf_ig import TF_IG
from neural_methods.trainers.base_trainer import BaseTrainer


class TF_IGTrainer(BaseTrainer):
    """Trainer for the TF-IG PPG → HR model."""

    def _model_dir_name(self):
        return f"tf_ig_{self.config.dataset_name}"

    def _build_model(self):
        return TF_IG(
            in_channels=getattr(self.config, "in_channels", 1),
            out_channels=getattr(self.config, "out_channels", 64),
            sampling_rate=self.sampling_rate,
        )

    def _forward_step(self, ppg, hr):
        """Multi-output forward: model returns (output, time_pred, freq_pred).
        Loss is computed on the fused output only."""
        model_output = self.model(ppg)

        if isinstance(model_output, (tuple, list)):
            hr_pred = model_output[0]
        else:
            hr_pred = model_output

        if hr_pred.dim() > 1:
            hr_pred = hr_pred.squeeze(-1)

        loss = self.criterion(hr_pred, hr)
        return loss, hr_pred