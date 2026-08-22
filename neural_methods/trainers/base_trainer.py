"""Base trainer providing the complete training/evaluation pipeline.

Subclasses only need to implement:
    _build_model()      → return the nn.Module
    _model_dir_name()   → return subdirectory name for checkpoints
    _forward_step()     → model-specific forward pass returning (loss, predictions)
"""
import os
import json
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from data_handler.bidmc_hanlder import BIDMC_Handler
from data_handler.capnobase_handler import CAPNOBASE_Handler
from utils.plot_prediction import plot_prediction


def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PPGWindowDataset(Dataset):
    """Wraps PPG windows + HR labels, resampling each window to a fixed length."""

    def __init__(self, records, input_dim):
        self.records = records
        self.input_dim = input_dim

    def __len__(self):
        return len(self.records)

    def _resample(self, values):
        """Resample one PPG window to exactly `input_dim` samples via linear interpolation."""
        signal = np.asarray(values, dtype=np.float32).reshape(-1)

        if signal.size == 0:
            raise ValueError("Cannot build a training sample from an empty PPG window")
        if signal.size == self.input_dim:
            return signal
        if signal.size == 1:
            return np.full(self.input_dim, signal.item(), dtype=np.float32)

        source_grid = np.linspace(0.0, 1.0, num=signal.size, dtype=np.float32)
        target_grid = np.linspace(0.0, 1.0, num=self.input_dim, dtype=np.float32)
        return np.interp(target_grid, source_grid, signal).astype(np.float32)

    def __getitem__(self, index):
        record = self.records[index]
        ppg = torch.from_numpy(self._resample(record["PPG"]))
        hr = torch.tensor(float(record["HR"]), dtype=torch.float32)
        return ppg, hr


class BaseTrainer(ABC):
    """Abstract base trainer for PPG → HR regression models.

    Provides the full pipeline: data loading, subject-level splitting,
    training with early stopping, validation, testing, checkpointing,
    TensorBoard logging, and metric computation (MAE, MSE, RMSE, R2,
    Pearson r, Mean Error / Bias).

    Subclasses implement _build_model(), _model_dir_name(), and
    _forward_step() to specialize for different architectures.
    """

    def __init__(self, config):
        self.config = config
        self.mode = getattr(config, "mode", "train_and_test")

        set_seed(getattr(config, "seed", 42))

        self.model_dir = os.path.join("pretrained_models", self._model_dir_name())

        self.train_loader, self.val_loader, self.test_loader, self.sampling_rate = (
            self._build_dataloaders()
        )

        self.model = self._build_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss()

        if self.mode == "train_and_test":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=getattr(config, "weight_decay", 0.0),
            )

            self.scheduler = None
            if getattr(config, "lr_scheduler", None) == "plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=getattr(config, "lr_factor", 0.5),
                    patience=getattr(config, "lr_patience", 5),
                )

            self.grad_clip_norm = getattr(config, "grad_clip_norm", None)
            self.early_stop_patience = getattr(config, "early_stop_patience", None)

            self.writer = SummaryWriter(log_dir=os.path.join(self.model_dir, "logs"))
            self.best_val_rmse = float("inf")
            self.history_path = os.path.join(self.model_dir, "history.json")

        elif self.mode == "only_test":
            model_path = getattr(config, "model_path", "")
            if not model_path or not os.path.exists(model_path):
                raise ValueError(
                    "config.mode == 'only_test' requires config.model_path to point "
                    f"to an existing checkpoint (got: {model_path!r})."
                )
            self._load_model_weights(model_path)
            print(f"[inference] Loaded pretrained weights from {model_path}")

        else:
            raise ValueError(
                f"Unsupported config.mode: {self.mode!r}. "
                "Expected 'train_and_test' or 'only_test'."
            )

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Instantiate and return the model."""

    @abstractmethod
    def _model_dir_name(self) -> str:
        """Return the subdirectory name under pretrained_models/."""

    @abstractmethod
    def _forward_step(self, ppg, hr):
        """Run the model-specific forward pass.

        Returns:
            loss:        scalar tensor (MSE on the primary output)
            predictions: tensor of shape (B,) — the HR predictions to track
        """

    # ------------------------------------------------------------------
    # Data loading & subject-level splitting
    # ------------------------------------------------------------------

    def _get_handler(self):
        """Return the dataset handler for `config.dataset_name`."""
        dataset_name = self.config.dataset_name.lower()
        if dataset_name == "bidmc":
            return BIDMC_Handler(self.config.data_path)
        if dataset_name == "capnobase":
            return CAPNOBASE_Handler(self.config.data_path)
        raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")

    def _load_records(self):
        """Load windowed data, dropping windows with missing/NaN HR labels."""
        handler = self._get_handler()
        handler.get_raw_data_from_csv()
        records = handler.process_data(
            window_size=self.config.window_size,
            window_shift=self.config.window_shift,
        )

        clean_records = [
            r for r in records
            if r.get("HR") is not None and not np.isnan(r["HR"])
        ]
        dropped = len(records) - len(clean_records)
        if dropped:
            print(f"[data] Dropped {dropped}/{len(records)} windows with missing/NaN HR labels")
        return clean_records

    def _split_subjects(self, subject_ids):
        """Deterministically partition subject ids into train/val/test sets."""
        val_split = getattr(self.config, "val_split", 0.15)
        test_split = getattr(self.config, "test_split", 0.15)

        if val_split + test_split >= 1.0:
            raise ValueError(
                f"val_split ({val_split}) + test_split ({test_split}) must sum to < 1.0"
            )

        n = len(subject_ids)
        if n < 3:
            raise ValueError(
                f"Need at least 3 subjects for train/val/test split; found {n}."
            )

        rng = random.Random(self.config.seed)
        shuffled = subject_ids[:]
        rng.shuffle(shuffled)

        n_val = max(1, round(n * val_split))
        n_test = max(1, round(n * test_split))

        while n_val + n_test >= n:
            if n_val >= n_test:
                n_val -= 1
            else:
                n_test -= 1
            if n_val < 1 or n_test < 1:
                raise ValueError(
                    f"Not enough subjects ({n}) for non-empty train/val/test splits "
                    f"with val_split={val_split}, test_split={test_split}."
                )

        test_subjects = set(shuffled[:n_test])
        val_subjects = set(shuffled[n_test:n_test + n_val])
        train_subjects = set(shuffled[n_test + n_val:])
        return train_subjects, val_subjects, test_subjects

    def _save_split_info(self, train_subjects, val_subjects, test_subjects):
        """Persist subject split to split_info.json for reproducibility."""
        info = {
            "seed": self.config.seed,
            "train_subjects": sorted(train_subjects),
            "val_subjects": sorted(val_subjects),
            "test_subjects": sorted(test_subjects),
        }
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            path = os.path.join(self.model_dir, "split_info.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            print(f"[split] Subject split saved to {path}")
        except OSError as e:
            print(f"[split] Warning: could not save split_info.json ({e})")

    def _split_info_path(self):
        return os.path.join(self.model_dir, "split_info.json")

    def _load_split_info(self):
        """Load a previously saved subject split, if valid."""
        path = self._split_info_path()
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                split_info = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[split] Warning: could not load split_info.json ({e})")
            return None

        required_keys = {"train_subjects", "val_subjects", "test_subjects"}
        if not required_keys.issubset(split_info):
            print(f"[split] Warning: split_info.json is missing required keys at {path}")
            return None
        return split_info

    def _records_from_split_info(self, records, split_info):
        """Re-apply a previously saved subject split to the current records."""
        train_subjects = set(split_info["train_subjects"])
        val_subjects = set(split_info["val_subjects"])
        test_subjects = set(split_info["test_subjects"])

        subject_ids = {r["id"] for r in records}
        missing = (train_subjects | val_subjects | test_subjects) - subject_ids
        if missing:
            raise ValueError(
                "Saved split_info.json references missing subjects: "
                f"{sorted(missing)}"
            )

        train_records = [r for r in records if r["id"] in train_subjects]
        val_records = [r for r in records if r["id"] in val_subjects]
        test_records = [r for r in records if r["id"] in test_subjects]

        print(
            f"[split] Reused saved split -> "
            f"{len(train_subjects)} train ({len(train_records)} windows) / "
            f"{len(val_subjects)} val ({len(val_records)} windows) / "
            f"{len(test_subjects)} test ({len(test_records)} windows)"
        )
        return train_records, val_records, test_records

    def _split_by_subject(self, records, split_info=None):
        """Split records by subject id (not by window) to prevent data leakage."""
        if split_info is not None:
            return self._records_from_split_info(records, split_info)

        subject_ids = sorted({r["id"] for r in records})
        train_subjects, val_subjects, test_subjects = self._split_subjects(subject_ids)

        train_records = [r for r in records if r["id"] in train_subjects]
        val_records = [r for r in records if r["id"] in val_subjects]
        test_records = [r for r in records if r["id"] in test_subjects]

        print(
            f"[split] {len(train_subjects)} train ({len(train_records)} windows) / "
            f"{len(val_subjects)} val ({len(val_records)} windows) / "
            f"{len(test_subjects)} test ({len(test_records)} windows)"
        )
        self._save_split_info(train_subjects, val_subjects, test_subjects)
        return train_records, val_records, test_records

    def _build_dataloaders(self):
        """Load records, resolve subject split, wrap in DataLoaders.

        In only_test mode, requires an existing split_info.json to guarantee
        test subjects match the split used during training.
        """
        records = self._load_records()
        if not records:
            raise ValueError("No processed records were produced from the dataset")

        fs = records[0]["sampling_rate"]

        split_info = None
        if self.mode == "only_test":
            split_info = self._load_split_info()
            if split_info is None:
                raise RuntimeError(
                    f"[split] only_test mode requires an existing split_info.json at "
                    f"{self._split_info_path()}, so the test subjects match training. "
                    "Refusing to generate a fresh split."
                )

        train_records, val_records, test_records = self._split_by_subject(
            records, split_info=split_info
        )

        num_workers = getattr(self.config, "num_workers", 0)
        train_loader = DataLoader(
            PPGWindowDataset(train_records, self.config.input_dim),
            batch_size=self.config.batch_size, shuffle=True, num_workers=num_workers,
        )
        val_loader = DataLoader(
            PPGWindowDataset(val_records, self.config.input_dim),
            batch_size=self.config.batch_size, shuffle=False, num_workers=num_workers,
        )
        test_loader = DataLoader(
            PPGWindowDataset(test_records, self.config.input_dim),
            batch_size=self.config.batch_size, shuffle=False, num_workers=num_workers,
        )
        return train_loader, val_loader, test_loader, fs

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(y_true, y_pred):
        """Compute evaluation metrics globally over the full concatenated arrays.

        Must be called on the complete set of predictions/labels for a split —
        NOT averaged from per-batch values — because R2 and Pearson r are not
        batch-separable.

        Returns: dict with MAE, MSE, RMSE, R2, Pearson_r, Mean_Error (Bias).
        """
        y_true = np.asarray(y_true).ravel().astype(np.float64)
        y_pred = np.asarray(y_pred).ravel().astype(np.float64)

        mae = float(np.mean(np.abs(y_pred - y_true)))
        mse = float(np.mean((y_pred - y_true) ** 2))
        rmse = float(np.sqrt(mse))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

        pearson_r = (
            float(np.corrcoef(y_true, y_pred)[0, 1])
            if y_true.std() > 0.0 and y_pred.std() > 0.0
            else float("nan")
        )

        mean_error = float(np.mean(y_pred - y_true))

        return {
            "MAE": mae, "MSE": mse, "RMSE": rmse,
            "R2": r2, "Pearson_r": pearson_r, "Mean_Error": mean_error,
        }

    @staticmethod
    def _print_metrics(model_name, metrics):
        """Print metrics in a standardized format."""
        print(
            f"[{model_name}] "
            f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | "
            f"RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f} | "
            f"Pearson r: {metrics['Pearson_r']:.4f} | "
            f"Bias: {metrics['Mean_Error']:.4f}"
        )

    # ------------------------------------------------------------------
    # Epoch execution
    # ------------------------------------------------------------------

    def _run_epoch(self, loader, training=True, return_predictions=False):
        """Run one pass over `loader`, collecting all predictions for global
        metric computation at epoch end.

        Calls _forward_step() for the model-specific forward pass and loss.
        For training batches, predictions are captured before optimizer.step(),
        so train_* metrics are a running within-epoch estimate — not a clean
        single-snapshot evaluation like val/test metrics.

        Returns: metrics dict, or (metrics, predictions, true_values) if
        return_predictions=True.
        """
        self.model.train() if training else self.model.eval()

        total_loss_sum = 0.0
        total_count = 0
        pred_true_pairs = []

        for ppg, hr in loader:
            ppg = ppg.to(self.device)
            hr = hr.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                loss, predictions = self._forward_step(ppg, hr)

                if training:
                    loss.backward()
                    if self.grad_clip_norm:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                pred_true_pairs.append(
                    (predictions.detach().cpu().numpy(), hr.detach().cpu().numpy())
                )

            batch_size = ppg.size(0)
            total_loss_sum += loss.item() * batch_size
            total_count += batch_size

        if total_count == 0:
            raise RuntimeError("Dataloader produced zero samples for this epoch")

        all_predictions = np.concatenate([p for p, _ in pred_true_pairs])
        all_true_values = np.concatenate([t for _, t in pred_true_pairs])

        eval_metrics = self._compute_metrics(all_true_values, all_predictions)

        metrics = {
            "mse": eval_metrics["MSE"],
            "mae": eval_metrics["MAE"],
            "rmse": eval_metrics["RMSE"],
            "r2": eval_metrics["R2"],
            "pearson_r": eval_metrics["Pearson_r"],
            "mean_error": eval_metrics["Mean_Error"],
            "loss": total_loss_sum / total_count,
        }

        if return_predictions:
            return metrics, all_predictions, all_true_values
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch, train_metrics, val_metrics, path):
        """Save model + optimizer state and epoch metrics."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }, path)

    def _load_model_weights(self, path):
        """Load model weights from a checkpoint (inference-only)."""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict) else checkpoint
        )
        self.model.load_state_dict(state_dict)

    def load_checkpoint(self, path):
        """Load full checkpoint (model + optimizer) to resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Train with early stopping, logging to TensorBoard and history.json.
        Saves only the best-val-RMSE checkpoint."""
        if self.mode != "train_and_test":
            raise RuntimeError("train() requires config.mode == 'train_and_test'")

        history = []
        epochs_without_improvement = 0

        try:
            for epoch in range(1, self.config.epochs + 1):
                train_metrics = self._run_epoch(self.train_loader, training=True)
                val_metrics = self._run_epoch(self.val_loader, training=False)

                if self.scheduler is not None:
                    self.scheduler.step(val_metrics["mse"])

                history.append({
                    "epoch": epoch,
                    "train_mse": train_metrics["mse"],
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "train_r2": train_metrics["r2"],
                    "train_pearson_r": train_metrics["pearson_r"],
                    "train_mean_error": train_metrics["mean_error"],
                    "train_loss": train_metrics["loss"],
                    "val_mse": val_metrics["mse"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "val_pearson_r": val_metrics["pearson_r"],
                    "val_mean_error": val_metrics["mean_error"],
                    "val_loss": val_metrics["loss"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

                # TensorBoard
                self.writer.add_scalar("Loss/Train", train_metrics["mse"], epoch)
                self.writer.add_scalar("Loss/Val", val_metrics["mse"], epoch)
                self.writer.add_scalar("R2/Train", train_metrics["r2"], epoch)
                self.writer.add_scalar("R2/Val", val_metrics["r2"], epoch)
                self.writer.add_scalar("Pearson/Train", train_metrics["pearson_r"], epoch)
                self.writer.add_scalar("Pearson/Val", val_metrics["pearson_r"], epoch)
                self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

                print(
                    f"Epoch {epoch:03d} | "
                    f"Train -> MSE: {train_metrics['mse']:.4f} | "
                    f"RMSE: {train_metrics['rmse']:.4f} | "
                    f"MAE: {train_metrics['mae']:.4f} | "
                    f"R2: {train_metrics['r2']:.4f} | "
                    f"Pearson r: {train_metrics['pearson_r']:.4f} | "
                    f"Bias: {train_metrics['mean_error']:.4f}\n"
                    f"          | "
                    f"Val   -> MSE: {val_metrics['mse']:.4f} | "
                    f"RMSE: {val_metrics['rmse']:.4f} | "
                    f"MAE: {val_metrics['mae']:.4f} | "
                    f"R2: {val_metrics['r2']:.4f} | "
                    f"Pearson r: {val_metrics['pearson_r']:.4f} | "
                    f"Bias: {val_metrics['mean_error']:.4f}"
                )

                if val_metrics["rmse"] < self.best_val_rmse:
                    self.best_val_rmse = val_metrics["rmse"]
                    epochs_without_improvement = 0
                    best_ckpt_path = os.path.join(self.model_dir, "best_model.pth")
                    self._save_checkpoint(epoch, train_metrics, val_metrics, best_ckpt_path)
                    print(f"          | New best (val RMSE: {self.best_val_rmse:.4f}) -> {best_ckpt_path}")
                else:
                    epochs_without_improvement += 1

                print("-" * 65)

                with open(self.history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

                if self.early_stop_patience and epochs_without_improvement >= self.early_stop_patience:
                    print(
                        f"Early stopping after {epoch} epochs "
                        f"(no val improvement for {epochs_without_improvement} epochs)"
                    )
                    break
        finally:
            self.writer.close()

        return history

    # ------------------------------------------------------------------
    # Testing & reporting
    # ------------------------------------------------------------------

    def test(self, save_plot=True, save_report=True):
        """Evaluate on held-out test set. Reloads best checkpoint in
        train_and_test mode; uses loaded weights in only_test mode."""
        if self.mode == "train_and_test":
            best_ckpt_path = os.path.join(self.model_dir, "best_model.pth")
            if os.path.exists(best_ckpt_path):
                self._load_model_weights(best_ckpt_path)
                print(f"[test] Loaded best checkpoint from {best_ckpt_path}")

        metrics, predictions, true_values = self._run_epoch(
            self.test_loader, training=False, return_predictions=True,
        )

        self._print_metrics(
            self._model_dir_name(),
            {
                "MAE": metrics["mae"], "MSE": metrics["mse"],
                "RMSE": metrics["rmse"], "R2": metrics["r2"],
                "Pearson_r": metrics["pearson_r"], "Mean_Error": metrics["mean_error"],
            },
        )

        if save_plot:
            os.makedirs(self.model_dir, exist_ok=True)
            plot_path = os.path.join(self.model_dir, "eval_predictions_test.png")
            plot_prediction(predictions, true_values, save_path=plot_path)
            print(f"[test] Prediction plot saved to {plot_path}")

        if save_report:
            self._save_final_report(metrics)

        return metrics

    def _save_final_report(self, test_metrics):
        """Write test metrics to test_report.json."""
        report = {"test_metrics": test_metrics}
        if self.mode == "train_and_test":
            report["best_val_rmse"] = self.best_val_rmse
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            report_path = os.path.join(self.model_dir, "test_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[report] Test report saved to {report_path}")
        except OSError as e:
            print(f"[report] Warning: could not save test_report.json ({e})")

    def run(self):
        """Single entry point: train+test or test-only based on config.mode."""
        if self.mode == "train_and_test":
            self.train()
            print("=" * 65)
            return self.test()
        else:
            return self.test()

    def save_model(self, path):
        """Save raw model state_dict."""
        torch.save(self.model.state_dict(), path)
