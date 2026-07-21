import pandas as pd

class BIDMC_Handler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.sampling_rate = 125  # Default sampling rate for BIDMC dataset
    

    def get_raw_data_from_csv(self):
        """ Load data from the specified path. """
        files = ["Breaths", "Numerics", "Signals"]
        num_of_subjects = 53
        
        data_list = []
        
        for subject_id in range(1, num_of_subjects + 1):
            subject_data = {"id": f"bidmc_{subject_id:02d}"}
            for file in files:
                file_path = f"{self.data_path}/bidmc_{subject_id:02d}_{file}.csv"
                df = pd.read_csv(file_path)
                
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()
                
                subject_data[file] = df
            data_list.append(subject_data)
            
        self.data = data_list
        return self.data
        

    def process_data(self, window_size=8, window_shift=2):
        """ Format of returned data:
            [
                {
                    "id": "bidmc_01",
                    "start": 0,
                    "end": 8,
                    "PPG": [0.43597, 0.43206, 0.42815, "...(125 samples)..."],
                    "HR": 94
                },
                {
                    "id": "bidmc_01",
                    "start": 2,
                    "end": 10,
                    "PPG": [0.58162, 0.58749, 0.59000, "...(125 samples)..."],
                    "HR": 94
                }
            ]
            
            - id: ID of the subject
            - start: Start time of the segment (in seconds)
            - end: End time of the segment (in seconds)
            - PPG: List of PPG values for the segment (125 samples)
            - HR: Heart rate for the segment (in beats per minute)
        """
        processed_data = []

        for subject_data in self.data:
            subject_id = subject_data["id"]
            signals_df = subject_data["Signals"]
            numerics_df = subject_data["Numerics"]

            ppg_signal = signals_df["PLETH"].values
            hr_values = numerics_df["HR"].values

            total_seconds = len(ppg_signal) // self.sampling_rate
            
            for start_time in range(0, total_seconds - window_size + 1, window_shift):
                end_time = start_time + window_size

                start_idx = start_time * self.sampling_rate
                end_idx = end_time * self.sampling_rate

                processed_data.append({
                    "id": subject_id,
                    "start": start_time,
                    "end": end_time,
                    "PPG": ppg_signal[start_idx:end_idx].tolist(),
                    "HR": hr_values[end_time - 1] if end_time - 1 < len(hr_values) else None,
                })

        return processed_data