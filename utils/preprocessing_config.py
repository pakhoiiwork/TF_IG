"""Shared preprocessing configuration for the STAG-HR dual-rate design.

The STAG-HR pipeline intentionally uses two different rate parameters for a
16-second PPG window:

1. TARGET_SAMPLING_RATE_HZ = 62.5
   Every PPG window is resampled to this rate before being handed to any model.
   For a window_size-second window, the model receives exactly
   round(window_size * TARGET_SAMPLING_RATE_HZ) samples. This is not arbitrary:
   it retains the HR band of interest (> 10 Hz Nyquist headroom), while the
   linear interpolation acts as an implicit anti-aliasing step that suppresses
   raw-rate motion artifact without discarding clinically relevant HR content.

2. dataset-native sampling_rate
   This remains the raw acquisition-rate value read from the dataset handler
   (for example, 125 Hz for BIDMC). It is used only by the STFT logic inside the
   frequency branch for native-resolution spectral analysis, not for the model's
   input vector length.

The important implication is that input_dim is a derived quantity:

    input_dim = round(window_size_seconds * TARGET_SAMPLING_RATE_HZ)

For the standard 16-second window, this yields 1000 samples, matching the
historical behavior without making the config file a free-form numeric knob.
See the rest of the pipeline for how this feeds the dataset resampling and the
frequency-domain branch.
"""

# [DUAL-RATE-DESIGN] Target rate every PPG window is resampled to before entering any model.
# See the module docstring for the full rationale; do not change this value without re-running
# the input-resolution ablation and confirming the native-rate STFT branch remains unchanged.
TARGET_SAMPLING_RATE_HZ = 62.5


def derive_input_dim(window_size_seconds: float) -> int:
    """Derive the fixed per-sample input length for a window_size-second PPG window.

    This must stay aligned with the dual-rate preprocessing design documented
    above: the dataset-native sampling rate remains separate from the model input
    rate, and the input tensor length is computed from the resample target rate.
    """
    return round(window_size_seconds * TARGET_SAMPLING_RATE_HZ)
