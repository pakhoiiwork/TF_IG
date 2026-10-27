import math
import torch
import torch.nn as nn

#             PPG
#              │
#   ┌──────────┴──────────┐
#   │                     │
# Time branch        STFT branch
#   │                     │
# Conv 7×1           Spectrogram
#   │                     │
# DW-Residual ×3    Lightweight CNN
# d = 1,2,4              │
#   │             Spectro-Temporal
# Temporal             Attention
# Attention               │
#   │                     │
#  Ft                    Ff
#   │                     │
#   └──────────┬──────────┘
#              │
#   Interaction Features
#    [Ft, Ff, Ft×Ff, |Ft-Ff|]
#              │
#        ┌─────┴─────┐
#        │           │
#   Interaction    Gate
#    projection      │
#        │    gFt+(1-g)Ff
#        └─────┬─────┘
#              │
#         Fused Feature
#              │
#          Fused HR


# ============================================================
# Utility blocks
# ============================================================

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution: reduces params from
    Cin*Cout*K to Cin*K + Cin*Cout."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class LightweightResidualBlock1d(nn.Module):
    """Residual block using depthwise-separable conv + 1×1 projection."""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + residual
        return self.act(out)


# ============================================================
# Temporal Attention Pooling
# ============================================================

class TemporalAttentionPooling(nn.Module):
    """Learned attention pooling over the temporal axis.
    Replaces global average pooling with a weighted sum where
    weights are produced by a small 1×1 conv attention network.

    Input:  (B, C, T)
    Output: pooled (B, C), weights (B, T)
    """

    def __init__(self, channels, hidden_channels=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        logits = self.attention(x)                      # (B, 1, T)
        weights = torch.softmax(logits, dim=-1)         # (B, 1, T)
        pooled = torch.sum(x * weights, dim=-1)         # (B, C)
        return pooled, weights.squeeze(1)


# ============================================================
# Time Domain Branch
# ============================================================

class TimeDomainBranch(nn.Module):
    """Lightweight time-domain PPG encoder.
    Stem (Conv7 → DWSep×2) followed by 3 dilated residual blocks
    and temporal attention pooling.

    Input:  (B, D)
    Output: features (B, C), attention (B, D)
    """

    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        mid_channels = max(16, out_channels // 2)

        # Initial morphology extractor
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv1d(mid_channels, out_channels, kernel_size=5, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=5, dilation=4),
        )

        self.block1 = LightweightResidualBlock1d(out_channels, kernel_size=7, dilation=1)
        self.block2 = LightweightResidualBlock1d(out_channels, kernel_size=5, dilation=2)
        self.block3 = LightweightResidualBlock1d(out_channels, kernel_size=3, dilation=4)

        self.attention_pool = TemporalAttentionPooling(
            channels=out_channels, hidden_channels=max(8, out_channels // 4),
        )

    def forward(self, x):
        x = x.unsqueeze(1)              # (B, 1, D)
        x = self.stem(x)                # (B, C, D)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        features, attention = self.attention_pool(x)
        return features, attention       # (B, C), (B, D)


# ============================================================
# Depthwise-Separable 2D Convolution
# ============================================================

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable 2D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False, dilation=dilation,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# ============================================================
# Spectro-Temporal Attention
# ============================================================

class SpectroTemporalAttention(nn.Module):
    """Joint attention over frequency and time axes of a 2D feature map.
    Produces a single attention map over (F, T) positions, then pools
    the feature map into a (B, C) vector via weighted sum.

    Input:  (B, C, F, T)
    Output: pooled (B, C), attention (B, F, T)
    """

    def __init__(self, channels, hidden_channels=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        B, C, Freq, Time = x.shape
        logits = self.attention(x)                              # (B, 1, F, T)
        weights = torch.softmax(logits.flatten(2), dim=-1)      # (B, 1, F*T)
        weights = weights.view(B, 1, Freq, Time)                # (B, 1, F, T)
        pooled = torch.sum(x * weights, dim=(2, 3))             # (B, C)
        return pooled, weights.squeeze(1)


# ============================================================
# Frequency Domain Branch
# ============================================================

class FrequencyDomainBranch(nn.Module):
    """STFT → lightweight 2D CNN → spectro-temporal attention.
    Converts the raw PPG signal to a magnitude spectrogram (log-scaled),
    masks it to the physiological HR range [min_freq, max_freq] Hz,
    then extracts features with a 2D stem + attention pooling.

    Input:  (B, D)
    Output: features (B, C), attention (B, F, T)

    """

    def __init__(self, sampling_rate, in_channels=1, out_channels=64,
                 min_freq=0.5, max_freq=5.0):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.win_length = int(sampling_rate * 8)
        self.hop_length = int(sampling_rate * 1)
        self.n_fft = 2 ** math.ceil(math.log2(self.win_length))

        self.register_buffer("window", torch.hann_window(self.win_length))

        mid_channels = max(16, out_channels // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(mid_channels, out_channels, kernel_size=5, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=5, dilation=4),
        )

        self.spectro_temporal_attention = SpectroTemporalAttention(
            channels=out_channels, hidden_channels=max(8, out_channels // 4),
        )

    def forward(self, x):
        # STFT → magnitude spectrogram
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window, return_complex=True,
        )  # (B, F, T)

        # Mask to physiological HR frequency range
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sampling_rate, device=x.device)
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        spec = spec[:, mask, :]                         # (B, F_selected, T)

        spec = torch.log1p(torch.abs(spec))             # Log-scaled magnitude
        spec = spec.unsqueeze(1)                        # (B, 1, F, T)

        features = self.stem(spec)                      # (B, C, F, T)
        features, attention = self.spectro_temporal_attention(features)
        return features, attention                      # (B, C), (B, F, T)


# ============================================================
# Interaction + Gated Fusion
# ============================================================

class InteractionGatedFusion(nn.Module):
    """Fuses time-domain and frequency-domain representations through
    explicit interaction features [t, f, t*f, |t-f|] and a learned
    feature-wise gate g*t + (1-g)*f.

    Input:  time_features (B, C), freq_features (B, C)
    Output: fused_features (B, C), gate (B, C)
    """

    def __init__(self, channels=64, dropout=0.2):
        super().__init__()
        # Project 4C interaction features back to C
        self.interaction = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Feature-wise reliability gate
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid(),
        )
        # Combine gated representation and interaction projection
        self.output = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, time_features, freq_features):
        # Explicit interaction features
        interaction_features = torch.cat([
            time_features, freq_features,
            time_features * freq_features,
            torch.abs(time_features - freq_features),
        ], dim=1)                                       # (B, 4C)
        interaction_features = self.interaction(interaction_features)  # (B, C)

        # Feature-wise gate
        gate = self.gate(torch.cat([time_features, freq_features], dim=1))  # (B, C)
        gated_features = gate * time_features + (1.0 - gate) * freq_features

        # Final fusion
        fused_features = self.output(
            torch.cat([gated_features, interaction_features], dim=1)
        )  # (B, C)
        return fused_features, gate


# ============================================================
# Regression Head
# ============================================================

class RegressionHead(nn.Module):
    """Two-layer MLP regression head: Linear → ReLU → Dropout → Linear(1)."""

    def __init__(self, in_channels=64, hidden_channels=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x):
        return self.head(x)


# ============================================================
# Final Model
# ============================================================

class TF_IG(nn.Module):
    """
        TF-IG: Lightweight dual-branch PPG heart rate estimation model.
    """

    def __init__(self, in_channels=1, out_channels=64, sampling_rate=125):
        super().__init__()

        self.time_branch = TimeDomainBranch(in_channels=in_channels, out_channels=out_channels)
        self.freq_branch = FrequencyDomainBranch(
            sampling_rate=sampling_rate, in_channels=in_channels, out_channels=out_channels,
        )
        self.fusion = InteractionGatedFusion(channels=out_channels, dropout=0.2)

        self.head = RegressionHead(in_channels=out_channels)

        # Learnable branch ensemble weights (softmax-normalized in forward)
        self.branch_logits = nn.Parameter(
            torch.tensor([0.6, 0.2, 0.2], dtype=torch.float32)
        )

    def forward(self, x, return_attention=False):
        # Time branch
        time_features, time_attention = self.time_branch(x)        # (B, C)

        # Frequency branch
        freq_features, spectro_temporal_attention = self.freq_branch(x)  # (B, C)

        # Fusion
        fused_features, fusion_gate = self.fusion(time_features, freq_features)

        # Regression heads (shared head for all branches)
        fused_pred = self.head(fused_features).squeeze(-1)         # (B,)
        time_pred = self.head(time_features).squeeze(-1)           # (B,)
        freq_pred = self.head(freq_features).squeeze(-1)           # (B,)

        # Weighted ensemble
        branch_weights = torch.softmax(self.branch_logits, dim=0)
        output = (
            branch_weights[0] * fused_pred
            + branch_weights[1] * time_pred
            + branch_weights[2] * freq_pred
        )

        if return_attention:
            return {
                "output": output,
                "fused_pred": fused_pred,
                "time_pred": time_pred,
                "freq_pred": freq_pred,
                "time_attention": time_attention,
                "spectro_temporal_attention": spectro_temporal_attention,
                "fusion_gate": fusion_gate,
                "branch_weights": branch_weights,
            }

        return (output, time_pred, freq_pred)