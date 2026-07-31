from pathlib import Path

import numpy as np
import pandas as pd
from utils.bandpass_filter import bandpass_filter


class CAPNOBASE_Handler:
    """Loader / windower for the CapnoBase PPG dataset."""

    def __init__(self, data_path, eps=1e-8):
        self.data_path = Path(data_path)
        self.data = None
        self.sampling_rate = 300  # Fallback sampling rate
        self.eps = eps

    def _normalize_numeric_series(self, values):
        if isinstance(values, pd.Series):
            values = values.iloc[0] if len(values) == 1 else values.values
        if isinstance(values, str):
            return np.fromstring(values.replace(",", " ").strip(), sep=" ")
        return np.asarray(values, dtype=float)

    def _get_sampling_rate(self, param_df):
        for col in ("samplingrate_pleth", "samplingrate_ecg", "samplingrate_co2"):
            if col in param_df.columns:
                try:
                    return int(param_df[col].iloc[0])
                except (ValueError, TypeError, IndexError):
                    pass
        return self.sampling_rate

    def get_raw_data_from_csv(self):
        files = ["labels", "meta", "param", "reference", "signal", "SFresults"]
        subject_ids = sorted({f.name.split("_8min_")[0] for f in self.data_path.glob("*_8min_*.csv")})

        data_list = []
        for subject_id in subject_ids:
            subject_data = {"id": f"capnobase_{subject_id}"}
            for file in files:
                file_path = self.data_path / f"{subject_id}_8min_{file}.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        df.columns = df.columns.str.strip()
                        subject_data[file] = df
                    except Exception as exc:
                        print(f"[CapnoBase] Warning: failed to read {file_path.name} ({exc})")

            data_list.append(subject_data)

        self.data = data_list
        return data_list

    def process_data(self, window_size=8, window_shift=2):
        if not self.data:
            raise RuntimeError("No data loaded. Call get_raw_data_from_csv() first.")

        processed_data = []

        for subject_data in self.data:
            subject_id = subject_data["id"]

            if "signal" not in subject_data or "reference" not in subject_data:
                print(f"[CapnoBase] Warning: missing signal/reference for {subject_id}, skipping subject")
                continue

            signal_df = subject_data["signal"]
            reference_df = subject_data["reference"]

            if "pleth_y" not in signal_df.columns or "hr_ecg_y" not in reference_df.columns:
                print(f"[CapnoBase] Warning: missing required columns for {subject_id}, skipping subject")
                continue

            sampling_rate = self._get_sampling_rate(subject_data["param"]) if "param" in subject_data else self.sampling_rate

            ppg_signal = signal_df["pleth_y"].values
            hr_values = self._normalize_numeric_series(reference_df["hr_ecg_y"])
            hr_time = self._normalize_numeric_series(reference_df["hr_ecg_x"]) if "hr_ecg_x" in reference_df.columns else None

            total_seconds = len(ppg_signal) // sampling_rate

            for start_time in range(0, total_seconds - window_size + 1, window_shift):
                end_time = start_time + window_size

                start_idx = start_time * sampling_rate
                end_idx = end_time * sampling_rate

                win_hr = hr_values[(hr_time >= start_time) & (hr_time < end_time)] if hr_time is not None else hr_values[start_time:end_time]
                win_ppg = ppg_signal[start_idx:end_idx]
                win_ppg = bandpass_filter(win_ppg, lowcut=0.5, highcut=5.0, fs=self.sampling_rate, order=4)
                win_ppg = (
                    win_ppg - np.mean(win_ppg)
                ) / (np.std(win_ppg) + self.eps)  # Normalize PPG window

                processed_data.append({
                    "id": subject_id,
                    "start": start_time,
                    "end": end_time,
                    "PPG": win_ppg.tolist(),
                    "HR": float(np.nanmean(win_hr)) if len(win_hr) > 0 else None,
                    "sampling_rate": sampling_rate,
                })

        return processed_data