"""Profile STAG-HR models with the same batch-1 protocol as MEDVSE."""

import argparse
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural_methods.models.baseline_2branch import Baseline2Branch
from neural_methods.models.baseline_mlp import BaselineModel
from neural_methods.models.baseline_transformer import BaselineTransformer
from neural_methods.models.stag_hr import Improved2Branch


def _make_model(name, sequence_length, sampling_rate):
    if name == "base":
        config = argparse.Namespace(hidden_dim1=512, hidden_dim2=256,
                                    hidden_dim3=64, output_dim=1)
        return BaselineModel(config, input_dim=sequence_length)
    if name == "base_trans":
        config = argparse.Namespace(patch_size=25, d_model=128, num_heads=2,
                                    ffn_dim=256, num_layers=2, output_dim=1)
        return BaselineTransformer(config, input_dim=sequence_length)
    if name == "base_2branch":
        return Baseline2Branch(in_channels=1, out_channels=64,
                               sampling_rate=sampling_rate)
    return Improved2Branch(in_channels=1, out_channels=64,
                           sampling_rate=sampling_rate)


def _model_size_bytes(model):
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def _profile_flops(model, sample):
    """Count profiler FLOPs; each MAC is reported separately as FLOPs / 2."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=False, with_flops=True) as profiler:
        model(sample)
    flops = sum(int(event.flops) for event in profiler.key_averages())
    return flops, flops // 2


def _peak_memory(model, sample, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        model(sample)
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return None


def _benchmark(model, sample, warmup, repeats, device):
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / repeats
    return latency_ms, 1000.0 / latency_ms


def _print_summary(name, checkpoint=None, sequence_length=1875,
                   sampling_rate=125, warmup=10, repeats=50, device_name=None):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _make_model(name, sequence_length, sampling_rate).to(device).eval()
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
    sample = torch.zeros((1, sequence_length), dtype=torch.float32, device=device)
    with torch.inference_mode():
        flops, macs = _profile_flops(model, sample)
        peak_memory = _peak_memory(model, sample, device)
        latency_ms, throughput = _benchmark(model, sample, warmup, repeats, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{name} (input=[1, {sequence_length}], device={device})")
    print(f"Params: {params:,}")
    print(f"FLOPs: {flops:,}")
    print(f"MACs: {macs:,}")
    print(f"Model size (MB): {_model_size_bytes(model) / (1024 ** 2):.3f}")
    print(f"Peak memory (MB): {peak_memory:.3f}" if peak_memory is not None else "Peak memory (MB): unavailable (CPU)")
    print(f"Inference time (ms/sample): {latency_ms:.3f}")
    print(f"Throughput (samples/s): {throughput:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "base_trans", "base_2branch", "base_i2branch"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sequence-length", type=int, default=1875,
                        help="Resampled BIDMC input length (30 s * 125 Hz * 0.5).")
    parser.add_argument("--sampling-rate", type=int, default=125)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    names = [args.model] if args.model else ["base", "base_trans", "base_2branch", "base_i2branch"]
    for name in names:
        _print_summary(name, args.checkpoint if args.model else None,
                       args.sequence_length, args.sampling_rate, args.warmup,
                       args.repeats, args.device)


if __name__ == "__main__":
    main()