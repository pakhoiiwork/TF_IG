import numpy as np
import pandas as pd


def _normalize_ecg_peaks(ecg_peak_x):
    if isinstance(ecg_peak_x, str):
        ecg_peak_x = ecg_peak_x.split()

    if isinstance(ecg_peak_x, pd.Series):
        ecg_peak_x = ecg_peak_x.tolist()

    if hasattr(ecg_peak_x, "tolist") and not isinstance(ecg_peak_x, list):
        ecg_peak_x = ecg_peak_x.tolist()

    if isinstance(ecg_peak_x, (list, tuple, np.ndarray)):
        ecg_peak_x = list(ecg_peak_x)
        if len(ecg_peak_x) == 1 and isinstance(ecg_peak_x[0], str):
            ecg_peak_x = ecg_peak_x[0].split()

    return np.asarray(ecg_peak_x, dtype=float)


def calculate_hr_from_ecg(ecg_peak_x, fs):
    """
    Calculate heart rate from ECG peak locations.

    Parameters:
    - ecg_peak_x: List or array of ECG peak locations (in samples).
    - fs: Sampling frequency (in Hz).

    Returns:
    - hr: Heart rate (in beats per minute).
    """
    ecg_peak_x = _normalize_ecg_peaks(ecg_peak_x)

    if ecg_peak_x.size < 2:
        return np.array([])

    # Calculate RR intervals in seconds
    rr_intervals = np.diff(ecg_peak_x) / fs
    
    # Calculate heart rate in beats per minute
    hr = 60 / rr_intervals
    
    return hr