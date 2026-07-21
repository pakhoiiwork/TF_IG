import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from neural_methods.models.baseline import BaselineModel
from data_handler.bidmc_hanlder import BIDMC_Handler
from data_handler.capnobase_handler import CAPNOBASE_Handler


def set_seed(seed: int) -> None:
    """Seed every RNG involved in the pipeline (python/numpy/torch/cuda) for reproducibility.

    `config.seed` previously only fed the random_split generator; DataLoader shuffling,
    weight init, and dropout were still non-deterministic run to run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PPGWindowDataset(Dataset):
    def __init__(self, records, input_dim):
        self.records = records
        self.input_dim = input_dim

    def __len__(self):
        return len(self.records)

    def _resample(self, values):
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


class BaselineTrainer:
    def __init__(self, config):
        self.config = config

        set_seed(getattr(config, "seed", 42))

        self.model = BaselineModel(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=getattr(config, "weight_decay", 0.0),
        )

        # Optional LR scheduler. Off by default (backward compatible with old configs);
        # enable by setting config.lr_scheduler = "plateau"
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

        self.train_loader, self.val_loader = self._build_dataloaders()

        self.model_dir = os.path.join("pretrained_models", f"baseline_{config.dataset_name}")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.model_dir, "logs"))
        self.best_val_rmse = float("inf")
        self.history_path = os.path.join(self.model_dir, "history.json")

    def _get_handler(self):
        dataset_name = self.config.dataset_name.lower()

        if dataset_name == "bidmc":
            return BIDMC_Handler(self.config.data_path)

        if dataset_name == "capnobase":
            return CAPNOBASE_Handler(self.config.data_path)

        raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")

    def _load_records(self):
        handler = self._get_handler()
        handler.get_raw_data_from_csv()
        records = handler.process_data(
            window_size=self.config.window_size,
            window_shift=self.config.window_shift,
        )

        # Some windows come back with HR=None (or NaN) when the numerics slice for
        # that window had no valid samples. Left as-is, `float(record["HR"])` in
        # PPGWindowDataset raises TypeError mid-training on whichever batch hits it.
        clean_records = [
            r for r in records
            if r.get("HR") is not None and not np.isnan(r["HR"])
        ]
        dropped = len(records) - len(clean_records)
        if dropped:
            print(f"[data] Dropped {dropped}/{len(records)} windows with missing/NaN HR labels")

        return clean_records

    def _split_by_subject(self, records):
        """Split into train/val by subject id rather than by individual window.

        `window_shift` is usually much smaller than `window_size` (e.g. 2s shift on
        an 8s window = 75% overlap), so consecutive windows from the same recording
        are near-duplicates. Splitting windows directly (the old `random_split`
        approach) routinely puts near-identical windows on both sides of the split,
        so the model is partly evaluated on signal it has already seen -> val
        metrics look better than true generalization to an unseen subject. Splitting
        by subject first removes that leakage.
        """
        subject_ids = sorted({r["id"] for r in records})
        if len(subject_ids) < 2:
            raise ValueError(
                "Need at least 2 subjects to create a subject-level train/val split; "
                f"found {len(subject_ids)}."
            )

        rng = random.Random(self.config.seed)
        shuffled_subjects = subject_ids[:]
        rng.shuffle(shuffled_subjects)

        val_subject_count = max(1, round(len(shuffled_subjects) * self.config.val_split))
        val_subject_count = min(val_subject_count, len(shuffled_subjects) - 1)  # keep >=1 train subject

        val_subjects = set(shuffled_subjects[:val_subject_count])
        train_subjects = set(shuffled_subjects[val_subject_count:])

        train_records = [r for r in records if r["id"] in train_subjects]
        val_records = [r for r in records if r["id"] in val_subjects]

        print(
            f"[split] {len(train_subjects)} train subjects ({len(train_records)} windows) / "
            f"{len(val_subjects)} val subjects ({len(val_records)} windows)"
        )

        return train_records, val_records

    def _build_dataloaders(self):
        records = self._load_records()
        if not records:
            raise ValueError("No processed records were produced from the dataset")

        train_records, val_records = self._split_by_subject(records)

        train_dataset = PPGWindowDataset(train_records, self.config.input_dim)
        val_dataset = PPGWindowDataset(val_records, self.config.input_dim)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 0),
        )

        return train_loader, val_loader

    def _run_epoch(self, loader, training=True):
        self.model.train() if training else self.model.eval()

        total_mse = 0.0
        total_mae = 0.0
        total_count = 0

        for ppg, hr in loader:
            ppg = ppg.to(self.device)
            hr = hr.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                predictions = self.model(ppg)
                if predictions.dim() > 1:
                    predictions = predictions.squeeze(-1)

                mse_loss = self.criterion(predictions, hr)
                mae_loss = torch.mean(torch.abs(predictions - hr))

                if training:
                    mse_loss.backward()
                    if self.grad_clip_norm:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

            batch_size = ppg.size(0)
            total_mse += mse_loss.item() * batch_size
            total_mae += mae_loss.item() * batch_size
            total_count += batch_size

        if total_count == 0:
            raise RuntimeError("Dataloader produced zero samples for this epoch")

        avg_mse = total_mse / total_count
        avg_mae = total_mae / total_count
        avg_rmse = float(np.sqrt(avg_mse))

        return {"mse": avg_mse, "mae": avg_mae, "rmse": avg_rmse}

    def _save_checkpoint(self, epoch, train_metrics, val_metrics, path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }, path)

    def load_checkpoint(self, path):
        """Resume training/inference from a checkpoint saved by `_save_checkpoint`."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

    def train(self):
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
                    "val_mse": val_metrics["mse"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

                self.writer.add_scalar("Loss/Train_MSE", train_metrics["mse"], epoch)
                self.writer.add_scalar("Loss/Train_MAE", train_metrics["mae"], epoch)
                self.writer.add_scalar("Loss/Train_RMSE", train_metrics["rmse"], epoch)
                self.writer.add_scalar("Loss/Val_MSE", val_metrics["mse"], epoch)
                self.writer.add_scalar("Loss/Val_MAE", val_metrics["mae"], epoch)
                self.writer.add_scalar("Loss/Val_RMSE", val_metrics["rmse"], epoch)
                self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

                print(
                    f"Epoch {epoch:03d} | "
                    f"Train -> MSE: {train_metrics['mse']:.4f} | RMSE: {train_metrics['rmse']:.4f} | MAE: {train_metrics['mae']:.4f}\n"
                    f"          | "
                    f"Val   -> MSE: {val_metrics['mse']:.4f} | RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f}"
                )

                # --- 1 checkpoint per epoch, as requested ---
                epoch_ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pth")
                self._save_checkpoint(epoch, train_metrics, val_metrics, epoch_ckpt_path)

                # --- rolling pointer to the latest epoch, for easy resuming ---
                last_ckpt_path = os.path.join(self.checkpoint_dir, "last.pth")
                self._save_checkpoint(epoch, train_metrics, val_metrics, last_ckpt_path)

                # --- best model tracked separately by val RMSE ---
                if val_metrics["rmse"] < self.best_val_rmse:
                    self.best_val_rmse = val_metrics["rmse"]
                    epochs_without_improvement = 0
                    best_ckpt_path = os.path.join(self.model_dir, "best_model.pth")
                    self._save_checkpoint(epoch, train_metrics, val_metrics, best_ckpt_path)
                    print(f"          | New best model (val RMSE: {self.best_val_rmse:.4f}) -> {best_ckpt_path}")
                else:
                    epochs_without_improvement += 1

                print("-" * 65)

                with open(self.history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

                if self.early_stop_patience and epochs_without_improvement >= self.early_stop_patience:
                    print(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(no val improvement for {epochs_without_improvement} epochs)"
                    )
                    break
        finally:
            self.writer.close()

        return history

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)