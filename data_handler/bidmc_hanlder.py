import pandas as pd

class BIDMC_Handler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None


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
        

    def process_data(self):
        """ Format of returned data:
            [
                {
                    "id": "bidmc_01",
                    "start": 0,
                    "end": 1,
                    "PPG": [0.43597, 0.43206, 0.42815, "...(125 samples)..."],
                    "HR": 94
                },
                {
                    "id": "bidmc_01",
                    "start": 1,
                    "end": 2,
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
            
            ppg_signal = signals_df['PLETH'].values
            hr_values = numerics_df['HR'].values
            
            # Segment the data into 1-second intervals (125 samples per second)
            num_segments = len(ppg_signal) // 125
            
            for i in range(num_segments):
                start_idx = i * 125
                end_idx = start_idx + 125
                
                segment_ppg = ppg_signal[start_idx:end_idx].tolist()
                segment_hr = hr_values[i] if i < len(hr_values) else None
                
                segment_data = {
                    "id": subject_id,
                    "start": i,
                    "end": i + 1,
                    "PPG": segment_ppg,
                    "HR": segment_hr
                }
                
                processed_data.append(segment_data)
        
        return processed_data