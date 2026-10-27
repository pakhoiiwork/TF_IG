from pathlib import Path

import numpy as np
import pandas as pd
from utils.bandpass_filter import bandpass_filter


class BIDMC_Handler:
    """Loader / windower for the BIDMC PPG-and-Respiration dataset."""

    def __init__(self, data_path, eps=1e-8):
        self.data_path = Path(data_path)
        self.data = None
        self.sampling_rate = 125  # PLETH (PPG) sampling rate for BIDMC
        self.eps = eps

    def get_raw_data_from_csv(self):
        """Load Breaths/Numerics/Signals CSVs for every subject found on disk."""
        files = ["Breaths", "Numerics", "Signals"]
        num_of_subjects = 53

        data_list = []

        for subject_id in range(1, num_of_subjects + 1):
            subject_key = f"bidmc_{subject_id:02d}"
            subject_data = {"id": subject_key}
            complete = True

            for file in files:
                file_path = self.data_path / f"{subject_key}_{file}.csv"

                if not file_path.exists():
                    print(f"[BIDMC] Warning: {file_path.name} not found, skipping subject {subject_key}")
                    complete = False
                    break

                try:
                    df = pd.read_csv(file_path)
                except Exception as exc:
                    print(f"[BIDMC] Warning: failed to read {file_path.name} ({exc}), skipping subject {subject_key}")
                    complete = False
                    break

                df.columns = df.columns.str.strip()
                subject_data[file] = df

            if complete:
                data_list.append(subject_data)

        self.data = data_list
        return self.data

    def process_data(self, window_size=8, window_shift=2):
        if not self.data:
            raise RuntimeError("No data loaded. Call get_raw_data_from_csv() first.")

        processed_data = []

        for subject_data in self.data:
            subject_id = subject_data["id"]
            signals_df = subject_data["Signals"]
            numerics_df = subject_data["Numerics"]

            if "PLETH" not in signals_df.columns:
                print(f"[BIDMC] Warning: 'PLETH' column missing for {subject_id}, skipping subject")
                continue
            if "HR" not in numerics_df.columns or "Time [s]" not in numerics_df.columns:
                print(f"[BIDMC] Warning: 'HR' or 'Time [s]' column missing for {subject_id}, skipping subject")
                continue

            ppg_signal = signals_df["PLETH"].values
            hr_values = numerics_df["HR"].values
            time_values = numerics_df["Time [s]"].values

            total_seconds = len(ppg_signal) // self.sampling_rate

            for start_time in range(0, total_seconds - window_size + 1, window_shift):
                end_time = start_time + window_size

                start_idx = start_time * self.sampling_rate
                end_idx = end_time * self.sampling_rate

                win_hr = hr_values[(time_values >= start_time) & (time_values < end_time)]
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
                    "sampling_rate": self.sampling_rate,
                })

        return processed_data