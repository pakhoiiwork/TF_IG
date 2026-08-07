import math
import torch
import torch.nn as nn

     #             PPG
    #                       │
    #        ┌──────────────┴──────────────┐
    #        │                             │
    #    Time branch                  STFT branch
    #        │                             │
    #    Conv 7×1                     Spectrogram
    #        │                             │
    #  DW-Residual ×3                Lightweight CNN
    #  d = 1,2,4                          │
    #        │                    Spectro-Temporal
    # Temporal Attention                Attention
    #        │                             │
    #       Ft                            Ff
    #        │                             │
    #        └──────────────┬──────────────┘
    #                       │
    #            Interaction Features
    #             [Ft, Ff, Ft×Ff,
    #                 |Ft-Ff|]
    #                       │
    #                 ┌─────┴─────┐
    #                 │           │
    #           Interaction     Gate
    #            projection       │
    #                 │     gFt+(1-g)Ff
    #                 └─────┬─────┘
    #                       │
    #                  Fused Feature
    #                       │
    #                   Fused HR



# ============================================================
# Utility blocks
# ============================================================

class DepthwiseSeparableConv1d(nn.Module):
    """
    Standard:
        Cin * Cout * K

    Depthwise separable:
        Cin * K + Cin * Cout
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
    ):
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class LightweightResidualBlock1d(nn.Module):
    """
    Lightweight temporal residual block
    """
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=1,
        dropout=0.0,
    ):
        super().__init__()

        self.conv1 = DepthwiseSeparableConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
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
        out = self.act(out)

        return out


# ============================================================
# V1: Temporal Attention Pooling
# ============================================================

class TemporalAttentionPooling(nn.Module):
    """
    Replaces global average pooling.

    Input:
        x: (B, C, T)

    Output:
        pooled:  (B, C)
        weights: (B, T)

    """
    def __init__(
        self,
        channels,
        hidden_channels=16,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(
                channels,
                hidden_channels,
                kernel_size=1,
            ),
            nn.Tanh(),
            nn.Conv1d(
                hidden_channels,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        # x: B,C,T

        logits = self.attention(x)
        # B,1,T

        weights = torch.softmax(
            logits,
            dim=-1,
        )

        pooled = torch.sum(
            x * weights,
            dim=-1,
        )
        # B,C

        return pooled, weights.squeeze(1)


# ============================================================
# Improved Time Domain Branch
# ============================================================

class TimeDomainBranch(nn.Module):
    """
    Lightweight time-domain PPG encoder.

    1. Residual blocks
    2. Dilated convolutions
    3. Depthwise-separable convolutions
    4. Temporal attention instead of GAP
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=64,
    ):
        super().__init__()

        # Initial local morphology extractor
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


        self.block1 = LightweightResidualBlock1d(
            out_channels,
            kernel_size=5,
            dilation=1,
        )

        self.block2 = LightweightResidualBlock1d(
            out_channels,
            kernel_size=5,
            dilation=2,
        )

        self.block3 = LightweightResidualBlock1d(
            out_channels,
            kernel_size=3,
            dilation=4,
        )

        self.attention_pool = TemporalAttentionPooling(
            channels=out_channels,
            hidden_channels=max(8, out_channels // 4),
        )

    def forward(self, x):
        """
        x:
            (B, D)

        return:
            features:
                (B, C)

            attention:
                (B, D)
        """

        x = x.unsqueeze(1)
        # B,1,D

        x = self.stem(x)
        # B,C,D

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        features, attention = self.attention_pool(x)

        return features, attention


# ============================================================
# Lightweight 2D convolution
# ============================================================

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)

        return x


# ============================================================
# V4: Spectro-Temporal Attention
# ============================================================

class SpectroTemporalAttention(nn.Module):
    """
    Joint attention over frequency and time.

    """
    def __init__(
        self,
        channels,
        hidden_channels=16,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        """
        x:
            B,C,F,T

        returns:
            pooled:
                B,C

            attention:
                B,F,T
        """

        B, C, Freq, Time = x.shape

        logits = self.attention(x)
        # B,1,F,T

        # Normalize jointly across all frequency-time positions.
        weights = torch.softmax(
            logits.flatten(start_dim=2),
            dim=-1,
        )

        weights = weights.view(
            B,
            1,
            Freq,
            Time,
        )

        pooled = torch.sum(
            x * weights,
            dim=(2, 3),
        )
        # B,C

        return pooled, weights.squeeze(1)


# ============================================================
# Frequency Domain Branch
# ============================================================

class FrequencyDomainBranch(nn.Module):
    """
    STFT -> lightweight 2D CNN -> spectro-temporal attention.
    """
    def __init__(
        self,
        sampling_rate,
        in_channels=1,
        out_channels=64,
        min_freq=0.5,
        max_freq=5.0,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.min_freq = min_freq
        self.max_freq = max_freq


        self.win_length = int(
            sampling_rate * 8
        )

        self.hop_length = int(
            sampling_rate * 1
        )

        self.n_fft = 2 ** math.ceil(
            math.log2(self.win_length)
        )

        self.register_buffer(
            "window",
            torch.hann_window(
                self.win_length
            ),
        )


        mid_channels = max(
            16,
            out_channels // 2,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
            ),

            DepthwiseSeparableConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
            ),
        )

        self.spectro_temporal_attention = (
            SpectroTemporalAttention(
                channels=out_channels,
                hidden_channels=max(
                    8,
                    out_channels // 4,
                ),
            )
        )

    def forward(self, x):
        """
        x:
            B,D

        returns:
            features:
                B,C

            attention:
                B,F,T
        """

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        # B,F,T

        freqs = torch.fft.rfftfreq(
            self.n_fft,
            d=1.0 / self.sampling_rate,
            device=x.device,
        )

        mask = (
            (freqs >= self.min_freq)
            & (freqs <= self.max_freq)
        )

        spec = spec[:, mask, :]
        # B,F_selected,T

        # Magnitude spectrogram
        spec = torch.abs(spec)

        # Recommended:
        spec = torch.log1p(spec)

        spec = spec.unsqueeze(1)
        # B,1,F,T

        features = self.stem(spec)
        # B,C,F,T

        features, attention = (
            self.spectro_temporal_attention(
                features
            )
        )

        return features, attention


# ============================================================
# V2: Interaction + Gated Fusion
# ============================================================

class InteractionGatedFusion(nn.Module):
    """
    Lightweight fusion between time-domain and
    frequency-domain representations.

    Explicit interaction:
        t
        f
        t * f
        |t - f|

    Then learn a feature-wise gate:
        g * t + (1-g) * f
    """
    def __init__(
        self,
        channels=64,
        dropout=0.2,
    ):
        super().__init__()

        # Project explicit interactions back to C dimensions.
        self.interaction = nn.Sequential(
            nn.Linear(
                channels * 4,
                channels,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Feature-wise reliability gate.
        self.gate = nn.Sequential(
            nn.Linear(
                channels * 2,
                channels,
            ),
            nn.Sigmoid(),
        )

        # Combine gated representation and interactions.
        self.output = nn.Sequential(
            nn.Linear(
                channels * 2,
                channels,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        time_features,
        freq_features,
    ):
        """
        time_features:
            B,C

        freq_features:
            B,C
        """

        interaction_features = torch.cat(
            [
                time_features,
                freq_features,
                time_features * freq_features,
                torch.abs(
                    time_features
                    - freq_features
                ),
            ],
            dim=1,
        )
        # B,4C

        interaction_features = self.interaction(
            interaction_features
        )
        # B,C

        # Feature-specific fusion weight.
        gate = self.gate(
            torch.cat(
                [
                    time_features,
                    freq_features,
                ],
                dim=1,
            )
        )
        # B,C

        gated_features = (
            gate * time_features
            + (1.0 - gate) * freq_features
        )

        fused_features = self.output(
            torch.cat(
                [
                    gated_features,
                    interaction_features,
                ],
                dim=1,
            )
        )

        return fused_features, gate


# ============================================================
# Regression Heads
# ============================================================

class RegressionHead(nn.Module):
    def __init__(
        self,
        in_channels=64,
        hidden_channels=32,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(
                in_channels,
                hidden_channels,
            ),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),

            nn.Linear(
                hidden_channels,
                1,
            ),
        )

    def forward(self, x):
        return self.head(x)


# ============================================================
# Final model
# ============================================================

class Improved2Branch(nn.Module):
    """
    Lightweight PPG HR estimation model.

    Improvements over Baseline2Branch:

    V1
        Global Average Pooling
        -> learned temporal attention pooling

    V2
        Concatenation fusion
        -> interaction + gated fusion

    V4
        frequency averaging
        -> joint spectro-temporal attention
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=64,
        sampling_rate=125,
    ):
        super().__init__()

        self.time_branch = TimeDomainBranch(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.freq_branch = FrequencyDomainBranch(
            sampling_rate=sampling_rate,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.fusion = InteractionGatedFusion(
            channels=out_channels,
            dropout=0.2,
        )

        # Separate heads because each representation
        # comes from a different feature space.
        self.fused_head = RegressionHead(
            in_channels=out_channels,
        )

        self.time_head = RegressionHead(
            in_channels=out_channels,
        )

        self.freq_head = RegressionHead(
            in_channels=out_channels,
        )

        # Keep your original ensemble idea but enforce
        # valid normalized weights.
        self.branch_logits = nn.Parameter(
            torch.tensor(
                [0.6, 0.2, 0.2],
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):
        # ------------------------------------------------
        # Time branch
        # ------------------------------------------------

        (
            time_features,
            time_attention,
        ) = self.time_branch(x)

        # B,C


        # ------------------------------------------------
        # Frequency branch
        # ------------------------------------------------

        (
            freq_features,
            spectro_temporal_attention,
        ) = self.freq_branch(x)

        # B,C


        # ------------------------------------------------
        # Fusion
        # ------------------------------------------------

        (
            fused_features,
            fusion_gate,
        ) = self.fusion(
            time_features,
            freq_features,
        )

        # ------------------------------------------------
        # Regression
        # ------------------------------------------------

        fused_pred = self.fused_head(
            fused_features
        ).squeeze(-1)

        time_pred = self.time_head(
            time_features
        ).squeeze(-1)

        freq_pred = self.freq_head(
            freq_features
        ).squeeze(-1)


        # ------------------------------------------------
        # Weighted ensemble
        # ------------------------------------------------

        branch_weights = torch.softmax(
            self.branch_logits,
            dim=0,
        )

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

                "spectro_temporal_attention":
                    spectro_temporal_attention,

                "fusion_gate": fusion_gate,

                "branch_weights": branch_weights,
            }

        # Keep interface similar to your original model.
        return (
            output,
            time_pred,
            freq_pred,
        )


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    sampling_rate = 125

    # Example:
    # 8-second PPG
    signal_length = (
        sampling_rate * 8
    )

    model = Improved2Branch(
        in_channels=1,
        out_channels=64,
        sampling_rate=sampling_rate,
    )

    x = torch.randn(
        8,
        signal_length,
    )

    outputs = model(
        x,
        return_attention=True,
    )

    print(
        "Output:",
        outputs["output"].shape
    )

    print(
        "Time prediction:",
        outputs["time_pred"].shape
    )

    print(
        "Frequency prediction:",
        outputs["freq_pred"].shape
    )

    print(
        "Time attention:",
        outputs["time_attention"].shape
    )

    print(
        "Spectro-temporal attention:",
        outputs[
            "spectro_temporal_attention"
        ].shape
    )

    print(
        "Fusion gate:",
        outputs["fusion_gate"].shape
    )

    print(
        "Branch weights:",
        outputs["branch_weights"]
    )


    # Parameter count
    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{num_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{num_params / 1e6:.3f} M"
    )