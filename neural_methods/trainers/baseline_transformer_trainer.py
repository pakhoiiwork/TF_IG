from neural_methods.models.baseline_transformer import BaselineTransformer
from neural_methods.trainers.base_trainer import BaseTrainer


class BaselineTransformerTrainer(BaseTrainer):
    """Trainer for the patch-based transformer baseline PPG → HR model."""

    def _model_dir_name(self):
        return f"baseline_transformer_{self.config.dataset_name}"

    def _build_model(self):
        return BaselineTransformer(self.config)

    def _forward_step(self, ppg, hr):
        """Single-output forward: model returns (B,) predictions directly."""
        predictions = self.model(ppg)
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        loss = self.criterion(predictions, hr)
        return loss, predictions