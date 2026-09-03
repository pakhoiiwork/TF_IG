from neural_methods.models.stag_hr import Improved2Branch
from neural_methods.trainers.base_trainer import BaseTrainer


class StagHrTrainer(BaseTrainer):
    """Trainer for the STAG-HR (Improved2Branch) PPG → HR model."""

    def _model_dir_name(self):
        return f"improved_2branch_{self.config.dataset_name}"

    def _build_model(self):
        # [DUAL-RATE-DESIGN] Native dataset sampling rate is intentionally kept here.
        return Improved2Branch(sampling_rate=self.sampling_rate)

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