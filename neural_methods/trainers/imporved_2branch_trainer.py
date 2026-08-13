import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from neural_methods.models.base_i2branch import Improved2Branch as Baseline2Branch
# from neural_methods.models.improved_2branch import Improved2Branch as Baseline2Branch
from neural_methods.loss.hrauxcon import HRAuxConLoss
from data_handler.bidmc_hanlder import BIDMC_Handler
from data_handler.capnobase_handler import CAPNOBASE_Handler

from utils.plot_prediction import plot_prediction


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch (and CUDA, if available) RNGs for reproducibility."""
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


class Improved2BranchTrainer:
    """
    Trains/evaluates a plain-MSE baseline PPG -> HR regressor.

    `config.mode`: "train_and_test" trains with early stopping (val is used
    only for best-checkpoint selection) then tests on held-out data;
    "only_test" loads a pretrained checkpoint and evaluates it directly.
    Subjects are split into disjoint train/val/test groups (never by window)
    so overlapping windows from the same recording can't leak across splits
    — see `_split_by_subject`.

    Metric computation (`_run_epoch`):
    - Single pass over the loader (train: forward+backward+step; val/test:
      forward only under no_grad) — no redundant second pass.
    - Every batch's predictions/labels are collected and concatenated at the
      end of the epoch; MAE/MSE/RMSE/R2/Pearson_r/Mean_Error are then computed
      ONCE, globally, via `_compute_metrics`. This matters because R2 and
      Pearson r are NOT separable across batches — averaging per-batch R2/Pearson
      values (weighted or not) gives a mathematically different, and wrong,
      number for the whole set. Only MAE/MSE (and therefore RMSE, taken as
      sqrt of the global MSE) are batch-summable; R2/Pearson require the full
      concatenated arrays regardless, so we compute everything from those
      arrays for consistency instead of duplicating the accumulation logic.
    - For TRAIN batches specifically, each batch's predictions are captured
      *before* that batch's `optimizer.step()` and while the model is in
      `train()` mode (so Dropout/BatchNorm behave as during training). This
      means `train_*` metrics reported per epoch are a "running" estimate
      (like Keras' on_batch metrics) accumulated across a moving set of
      weights within the epoch — not a clean end-of-epoch re-evaluation of a
      single frozen weight snapshot. This is intentional (matches the more
      efficient single-pass design) but should not be compared 1:1 against
      `val_*`/test metrics, which ARE a clean single-weight-snapshot
      evaluation (val/test never call `optimizer.step()`).
    """

    def __init__(self, config):
        """Build dataloaders and model, then set up training (optimizer/scheduler/
        logging) or inference-only (load pretrained weights), based on `config.mode`."""
        self.config = config
        self.mode = getattr(config, "mode", "train_and_test")

        set_seed(getattr(config, "seed", 42))

        self.model_dir = os.path.join("pretrained_models", f"improved_2branch_{config.dataset_name}")

        self.train_loader, self.val_loader, self.test_loader, self.sampling_rate = self._build_dataloaders()

        self.model = Baseline2Branch(sampling_rate=self.sampling_rate)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = HRAuxConLoss()

        if self.mode == "train_and_test":
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

            self.writer = SummaryWriter(log_dir=os.path.join(self.model_dir, "logs"))
            self.best_val_rmse = float("inf")
            self.history_path = os.path.join(self.model_dir, "history.json")

        elif self.mode == "only_test":
            # Inference-only: no optimizer/scheduler/writer needed.
            # Just load the pretrained weights so `test()` has something to score.
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

    def _get_handler(self):
        """Return the dataset handler (BIDMC or CapnoBase) for `config.dataset_name`."""
        dataset_name = self.config.dataset_name.lower()

        if dataset_name == "bidmc":
            return BIDMC_Handler(self.config.data_path)

        if dataset_name == "capnobase":
            return CAPNOBASE_Handler(self.config.data_path)

        raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")

    def _load_records(self):
        """Load raw data into windows, dropping any with missing/NaN HR labels."""
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

    def _split_subjects(self, subject_ids):
        """Deterministically partition subject ids into disjoint train/val/test
        sets, sized by `config.val_split`/`config.test_split` (default 0.15
        each), guaranteeing at least 1 subject per split."""
        val_split = getattr(self.config, "val_split", 0.15)
        test_split = getattr(self.config, "test_split", 0.15)

        if val_split + test_split >= 1.0:
            raise ValueError(
                f"val_split ({val_split}) + test_split ({test_split}) must sum to < 1.0"
            )

        n = len(subject_ids)
        if n < 3:
            raise ValueError(
                "Need at least 3 subjects to create a non-empty train/val/test "
                f"subject-level split; found {n}."
            )

        rng = random.Random(self.config.seed)
        shuffled = subject_ids[:]
        rng.shuffle(shuffled)

        n_val = max(1, round(n * val_split))
        n_test = max(1, round(n * test_split))

        # Shrink val/test (larger one first) until at least 1 subject is left for train.
        while n_val + n_test >= n:
            if n_val >= n_test:
                n_val -= 1
            else:
                n_test -= 1
            if n_val < 1 or n_test < 1:
                raise ValueError(
                    f"Not enough subjects ({n}) to carve out non-empty train/val/test "
                    f"splits with val_split={val_split}, test_split={test_split}."
                )

        test_subjects = set(shuffled[:n_test])
        val_subjects = set(shuffled[n_test:n_test + n_val])
        train_subjects = set(shuffled[n_test + n_val:])

        return train_subjects, val_subjects, test_subjects

    def _save_split_info(self, train_subjects, val_subjects, test_subjects):
        """Persist the train/val/test subject ids to `split_info.json`, so
        reported test metrics can always be traced back to exactly which
        subjects they were computed on."""
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
        """Path to this run's split_info.json."""
        return os.path.join(self.model_dir, "split_info.json")

    def _load_split_info(self):
        """Load a previously saved subject split, if one exists and is valid."""
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
        declared_subjects = train_subjects | val_subjects | test_subjects
        missing_subjects = declared_subjects - subject_ids

        if missing_subjects:
            raise ValueError(
                "Saved split_info.json references subjects that are not present in the current dataset: "
                f"{sorted(missing_subjects)}"
            )

        train_records = [r for r in records if r["id"] in train_subjects]
        val_records = [r for r in records if r["id"] in val_subjects]
        test_records = [r for r in records if r["id"] in test_subjects]

        print(
            f"[split] Reused saved split -> "
            f"{len(train_subjects)} train subjects ({len(train_records)} windows) / "
            f"{len(val_subjects)} val subjects ({len(val_records)} windows) / "
            f"{len(test_subjects)} test subjects ({len(test_records)} windows)"
        )

        return train_records, val_records, test_records

    def _split_by_subject(self, records, split_info=None):
        """Split records into train/val/test by subject id (not by window),
        since window_shift << window_size means consecutive windows from the
        same recording are near-duplicates and would leak across a window-level
        split. Reuses `split_info` if given, otherwise creates and saves a new split."""
        if split_info is not None:
            return self._records_from_split_info(records, split_info)

        subject_ids = sorted({r["id"] for r in records})
        train_subjects, val_subjects, test_subjects = self._split_subjects(subject_ids)

        train_records = [r for r in records if r["id"] in train_subjects]
        val_records = [r for r in records if r["id"] in val_subjects]
        test_records = [r for r in records if r["id"] in test_subjects]

        print(
            f"[split] {len(train_subjects)} train subjects ({len(train_records)} windows) / "
            f"{len(val_subjects)} val subjects ({len(val_records)} windows) / "
            f"{len(test_subjects)} test subjects ({len(test_records)} windows)"
        )

        self._save_split_info(train_subjects, val_subjects, test_subjects)

        return train_records, val_records, test_records

    def _build_dataloaders(self):
        """Load records, resolve the train/val/test subject split, and wrap
        each subset in a DataLoader.

        In `only_test` mode this requires an existing `split_info.json` — if
        it's missing, we hard-fail instead of silently generating a fresh
        split, since a mismatched split could evaluate the model on subjects
        it was actually trained on."""
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
                    f"{self._split_info_path()}, so the test subjects are guaranteed to "
                    "match the split used for training. Refusing to fall back to a "
                    "freshly generated split, since that could silently evaluate on "
                    "subjects the model was trained on."
                )

        train_records, val_records, test_records = self._split_by_subject(records, split_info=split_info)

        train_dataset = PPGWindowDataset(train_records, self.config.input_dim)
        val_dataset = PPGWindowDataset(val_records, self.config.input_dim)
        test_dataset = PPGWindowDataset(test_records, self.config.input_dim)

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
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 0),
        )

        return train_loader, val_loader, test_loader, fs

    def _compute_metrics(self, y_true, y_pred):
        """
        Compute evaluation metrics exactly like the TensorFlow evaluation code,
        GLOBALLY over the full (concatenated) arrays passed in. Must always be
        called on the complete set of predictions/labels for a split — never
        averaged/combined from per-batch metric values, since R2 and Pearson r
        are not batch-separable (their averages across batches are not equal
        to the value computed on the full set).

        Metrics:
            MAE
            MSE
            RMSE
            R2
            Pearson_r
            Mean_Error (Bias)
        """
        y_true = np.asarray(y_true).ravel().astype(np.float64)
        y_pred = np.asarray(y_pred).ravel().astype(np.float64)

        # MAE
        mae = float(np.mean(np.abs(y_pred - y_true)))

        # MSE
        mse = float(np.mean((y_pred - y_true) ** 2))

        # RMSE
        rmse = float(np.sqrt(mse))

        # R2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        r2 = (
            float(1.0 - ss_res / ss_tot)
            if ss_tot > 0.0
            else float("nan")
        )

        # Pearson R
        pearson_r = (
            float(np.corrcoef(y_true, y_pred)[0, 1])
            if y_true.std() > 0.0 and y_pred.std() > 0.0
            else float("nan")
        )

        # Bias / Mean Error
        mean_error = float(np.mean(y_pred - y_true))

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Pearson_r": pearson_r,
            "Mean_Error": mean_error,
        }

    def _print_metrics(self, model_name, metrics):
        """Print metrics in exactly the same format as the TensorFlow code."""
        print(
            "[" + model_name + "] "
            "MAE: {MAE:.4f} | MSE: {MSE:.4f} | RMSE: {RMSE:.4f} | "
            "R2: {R2:.4f} | Pearson r: {Pearson_r:.4f} | "
            "Bias (Mean Error): {Mean_Error:.4f}".format(**metrics)
        )

    def _run_epoch(self, loader, training=True, return_predictions=False):
        """Run ONE pass over `loader` (forward, plus backward+step if `training`).

        Every batch's predictions/labels are collected as we go, then
        concatenated once at the end to compute full-set metrics (MAE, MSE,
        RMSE, R2, Pearson_r, Mean_Error) via `_compute_metrics`. This avoids
        a second forward-only pass (unlike a naive two-pass implementation)
        while still being numerically correct for R2/Pearson r, which cannot
        be derived from per-batch averages.

        For TRAIN, predictions are captured pre-`optimizer.step()` for that
        batch, in `train()` mode — so `train_*` metrics are a running,
        within-epoch estimate over a moving set of weights, not a clean
        single-snapshot evaluation. VAL/TEST never call `optimizer.step()`,
        so their metrics ARE a clean single-snapshot evaluation.

        Returns `metrics` (dict), or `(metrics, predictions, true_values)`
        if `return_predictions=True`.
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
                model_output = self.model(ppg)

                if isinstance(model_output, (tuple, list)):
                    if len(model_output) != 3:
                        raise ValueError(
                            f"Expected 3 outputs from model, got {len(model_output)}"
                        )
                    hr_pred, hr_time, hr_freq = model_output
                else:
                    hr_pred = model_output
                    hr_time = model_output
                    hr_freq = model_output

                if hr_pred.dim() > 1:
                    hr_pred = hr_pred.squeeze(-1)
                if hr_time.dim() > 1:
                    hr_time = hr_time.squeeze(-1)
                if hr_freq.dim() > 1:
                    hr_freq = hr_freq.squeeze(-1)

                total_loss, loss_components = self.criterion(hr_pred, hr_time, hr_freq, hr)

                if training:
                    total_loss.backward()
                    if self.grad_clip_norm:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                # Always collect — needed for global MAE/MSE/RMSE/R2/Pearson/Bias,
                # not only when a caller wants the raw arrays back.
                pred_true_pairs.append(
                    (hr_pred.detach().cpu().numpy(), hr.detach().cpu().numpy())
                )

            batch_size = ppg.size(0)
            total_loss_sum += total_loss.item() * batch_size
            total_count += batch_size

        if total_count == 0:
            raise RuntimeError("Dataloader produced zero samples for this epoch")

        avg_total_loss = total_loss_sum / total_count

        all_predictions = np.concatenate([p for p, t in pred_true_pairs])
        all_true_values = np.concatenate([t for p, t in pred_true_pairs])

        evaluation_metrics = self._compute_metrics(all_true_values, all_predictions)

        metrics = {
            "mse": evaluation_metrics["MSE"],
            "mae": evaluation_metrics["MAE"],
            "rmse": evaluation_metrics["RMSE"],
            "r2": evaluation_metrics["R2"],
            "pearson_r": evaluation_metrics["Pearson_r"],
            "mean_error": evaluation_metrics["Mean_Error"],
            "total_loss": avg_total_loss,
        }

        if return_predictions:
            return metrics, all_predictions, all_true_values

        return metrics

    def _save_checkpoint(self, epoch, train_metrics, val_metrics, path):
        """Save model + optimizer state, plus this epoch's metrics, to `path`."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }, path)

    def _load_model_weights(self, path):
        """Load only the model weights from a checkpoint (no optimizer state),
        for inference-only evaluation. Handles both full checkpoint dicts saved
        by `_save_checkpoint` and raw state_dicts saved by `save_model`."""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)

    def load_checkpoint(self, path):
        """Load model + optimizer state from a full checkpoint (e.g. `best_model.pth`)
        to resume training; returns the epoch it was saved at."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

    def train(self):
        """Run the training loop: each epoch, train + validate, log to
        TensorBoard/history.json, and save `best_model.pth` whenever val RMSE
        improves (this is the only checkpoint kept — no per-epoch checkpoints).
        Stops early if `config.early_stop_patience` epochs pass with no
        val-RMSE improvement."""
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
                    "train_total_loss": train_metrics["total_loss"],
                    "val_mse": val_metrics["mse"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "val_pearson_r": val_metrics["pearson_r"],
                    "val_mean_error": val_metrics["mean_error"],
                    "val_total_loss": val_metrics["total_loss"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

                self.writer.add_scalar("Loss/Train", train_metrics["mse"], epoch)
                self.writer.add_scalar("Loss/Val", val_metrics["mse"], epoch)
                self.writer.add_scalar("Loss/Total/Train", train_metrics["total_loss"], epoch)
                self.writer.add_scalar("Loss/Total/Val", val_metrics["total_loss"], epoch)
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
                    f"Bias: {train_metrics['mean_error']:.4f} | "
                    f"Total Loss: {train_metrics['total_loss']:.4f}\n"
                    f"          | "
                    f"Val   -> MSE: {val_metrics['mse']:.4f} | "
                    f"RMSE: {val_metrics['rmse']:.4f} | "
                    f"MAE: {val_metrics['mae']:.4f} | "
                    f"R2: {val_metrics['r2']:.4f} | "
                    f"Pearson r: {val_metrics['pearson_r']:.4f} | "
                    f"Bias: {val_metrics['mean_error']:.4f} | "
                    f"Total Loss: {val_metrics['total_loss']:.4f}"
                )

                # Only the best-val-RMSE checkpoint is kept.
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

    def test(self, save_plot=True, save_report=True):
        """Evaluate on the held-out test set — the number to report, computed
        exactly like the TensorFlow evaluation code (MAE, MSE, RMSE, R2,
        Pearson r, Bias/Mean Error), globally over the full test set.

        In `train_and_test` mode, reloads `best_model.pth` first (independent
        of whatever weights are in memory); in `only_test` mode, scores
        whatever weights were already loaded from `config.model_path`."""
        if self.mode == "train_and_test":
            best_ckpt_path = os.path.join(self.model_dir, "best_model.pth")
            if os.path.exists(best_ckpt_path):
                self._load_model_weights(best_ckpt_path)
                print(f"[test] Loaded best checkpoint from {best_ckpt_path}")

        metrics, predictions, true_values = self._run_epoch(
            self.test_loader, training=False, return_predictions=True
        )

        self._print_metrics(
            "Improved2Branch",
            {
                "MAE": metrics["mae"],
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "R2": metrics["r2"],
                "Pearson_r": metrics["pearson_r"],
                "Mean_Error": metrics["mean_error"],
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
        """Write the final test metrics (and best val RMSE, if available) to
        `test_report.json` — this is the number that should go in a paper."""
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
        """Single entry point, dispatched on `config.mode`: "train_and_test"
        trains then tests on held-out data; "only_test" tests the loaded
        checkpoint directly. Either way, returns the test metrics."""
        if self.mode == "train_and_test":
            self.train()
            print("=" * 65)
            return self.test()
        else:  # only_test
            return self.test()

    def save_model(self, path):
        """Save just the raw model state_dict to `path`."""
        torch.save(self.model.state_dict(), path)