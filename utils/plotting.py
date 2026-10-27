from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


def plot_ppg(data, save_path=None):
    save_path = Path(save_path) if save_path else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    for subject_data in data:
        ppg_signal = subject_data["PPG"]
        time_axis = [i / subject_data["sampling_rate"] for i in range(len(ppg_signal))]

        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, ppg_signal, label="PPG Signal", color="blue")
        plt.title(
            f"PPG Signal for {subject_data['id']} "
            f"(Start: {subject_data['start']}s, End: {subject_data['end']}s)"
        )
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


def plot_prediction(predict_val, true_val, save_path=None):
    predict_val = np.asarray(predict_val).squeeze()
    true_val = np.asarray(true_val).squeeze()

    plt.figure(figsize=(12, 4))
    plt.plot(true_val, label="True Values", color="green", alpha=0.8)
    plt.plot(predict_val, label="Predicted Values", color="red", alpha=0.8)
    plt.title("Prediction vs True Values")
    plt.xlabel("Window Index")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == "":
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / "prediction_vs_true.png"
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            file_path = save_path
        if file_path.is_dir():
            shutil.rmtree(file_path)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"[plot] Saved plot to {file_path}")
    else:
        plt.show()
    plt.close()
