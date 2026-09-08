"""Shared preprocessing configuration."""

TARGET_SAMPLING_RATE_FRACTION = 0.5


def derive_target_sampling_rate(sampling_rate_hz: float) -> float:
    """Return the target rate as a fraction of the dataset sampling rate."""
    return sampling_rate_hz * TARGET_SAMPLING_RATE_FRACTION


def derive_input_dim(window_size_seconds: float, sampling_rate_hz: float) -> int:
    """Derive the per-sample input length for a PPG window."""
    return round(window_size_seconds * derive_target_sampling_rate(sampling_rate_hz))
