from neural_methods.models.baseline_mlp import BaselineModel
from neural_methods.trainers.base_trainer import BaseTrainer


class BaselineMlpTrainer(BaseTrainer):
    """Trainer for the MLP baseline PPG → HR model."""

    def _model_dir_name(self):
        return f"baseline_{self.config.dataset_name}"

    def _build_model(self):
        return BaselineModel(self.config, input_dim=self.input_dim)

    def _forward_step(self, ppg, hr):
        """Single-output forward: model returns (B,) predictions directly."""
        predictions = self.model(ppg)
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        loss = self.criterion(predictions, hr)
        return loss, predictions