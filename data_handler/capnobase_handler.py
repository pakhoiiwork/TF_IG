from pathlib import Path
from utils.calculate_hr import calculate_hr_from_ecg
import pandas as pd


class CAPNOBASE_Handler:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = None

    def _get_sampling_rate(self, param_df):
        for column in ("samplingrate_pleth", "samplingrate_ecg", "samplingrate_co2"):
            if column in param_df.columns:
                return int(param_df[column].iloc[0])
        raise KeyError("No sampling rate column found in CapnoBase param data")

    def get_raw_data_from_csv(self):
        files = ["labels", "meta", "param", "reference", "signal", "SFresults"]

        # Automatically get all subject IDs
        subject_ids = sorted({
            f.name.split("_8min_")[0]
            for f in self.data_path.glob("*_8min_*.csv")
        })

        # print(f"Found {len(subject_ids)} subjects.")

        data_list = []

        for subject_id in subject_ids:
            subject_data = {"id": f"capnobase_{subject_id}"}

            for file in files:
                file_path = self.data_path / f"{subject_id}_8min_{file}.csv"

                if not file_path.exists():
                    # print(f"Warning: {file_path.name} not found.")
                    continue

                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()

                subject_data[file] = df

            data_list.append(subject_data)

        self.data = data_list
        return data_list
    
    
    def process_data(self):
        """ Format of returned data:
            [
                {
                    "id": "capnobase_01",
                    "start": 0,
                    "end": 1,
                    "PPG": [0.43597, 0.43206, 0.42815, "...(300 samples)..."],
                    "HR": 94
                },
                {
                    "id": "capnobase_01",
                    "start": 1,
                    "end": 2,
                    "PPG": [0.58162, 0.58749, 0.59000, "...(300 samples)..."],
                    "HR": 94
                }
            ]
            
            - id: ID of the subject
            - start: Start time of the segment (in seconds)
            - end: End time of the segment (in seconds)
            - PPG: List of PPG values for the segment (300 samples)
            - HR: Heart rate for the segment (in beats per minute)
        """
        processed_data = []

        for subject_data in self.data:
            subject_id = subject_data["id"]
            signal_df = subject_data["signal"]
            labels_df = subject_data["labels"]
            param_df = subject_data["param"]
            
            ppg_signal = signal_df['pleth_y'].values
            ecg_peaks = labels_df['ecg_peak_x'].values

            sampling_rate = self._get_sampling_rate(param_df)
            hr_values = calculate_hr_from_ecg(ecg_peaks, sampling_rate)

            num_segments = len(ppg_signal) // sampling_rate
            # print(f"Processing subject {subject_id}: {num_segments} segments.")
            
            for i in range(num_segments):
                start_idx = i * sampling_rate
                end_idx = start_idx + sampling_rate

                segment_ppg = ppg_signal[start_idx:end_idx]
                segment_hr = hr_values[i] if i < len(hr_values) else None

                segment_data = {
                    "id": subject_id,
                    "start": i,
                    "end": i + 1,
                    "PPG": segment_ppg.tolist(),
                    "HR": segment_hr
                }
                
                processed_data.append(segment_data)

        return processed_data
            