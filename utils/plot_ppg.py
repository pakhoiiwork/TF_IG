from pathlib import Path
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_handler.bidmc_hanlder import BIDMC_Handler
from data_handler.capnobase_handler import CAPNOBASE_Handler

def plot_ppg(data, save_path=None):
    save_path = Path(save_path) if save_path else None
    save_path.mkdir(parents=True, exist_ok=True) if save_path else None
    
    for subject_data in data:
        ppg_signal = subject_data["PPG"]
        time_axis = [i / subject_data["sampling_rate"] for i in range(len(ppg_signal))]

        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, ppg_signal, label="PPG Signal", color='blue')
        plt.title(f"PPG Signal for {subject_data['id']} (Start: {subject_data['start']}s, End: {subject_data['end']}s)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("PPG Value")
        plt.legend()
        plt.grid()
        
        file_name = f"{subject_data['id']}_start{subject_data['start']}_end{subject_data['end']}.png"
        if save_path:
            plt.savefig(save_path / file_name)
        else:
            plt.show()
            
        plt.close()

def main():
    bidmc_path = ROOT / "datasets" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
    capnobase_path = ROOT / "datasets" / "CapnoBase" / "data" / "csv"

    bidmc_handler = BIDMC_Handler(str(bidmc_path))
    capno_handler = CAPNOBASE_Handler(capnobase_path)

    bidmc_handler.get_raw_data_from_csv()
    capno_handler.get_raw_data_from_csv()

    bidmc_processed = bidmc_handler.process_data()
    capno_processed = capno_handler.process_data()
    
    save_path_bidmc = ROOT / "plots" / "bidmc"
    save_path_capnobase = ROOT / "plots" / "capnobase"
    
    plot_ppg(bidmc_processed, save_path=save_path_bidmc)
    plot_ppg(capno_processed, save_path=save_path_capnobase)


if __name__ == "__main__":
    main()