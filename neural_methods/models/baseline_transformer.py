import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Projects fixed-size patches of the input signal into d_model dimensions."""

    def __init__(self, input_dim, patch_size=25, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x):
        # x: (B, input_dim)
        B, D = x.shape
        x = x.view(B, self.num_patches, self.patch_size)  # (B, num_patches, patch_size)
        x = self.proj(x)  # (B, num_patches, d_model)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for transformer patches."""

    def __init__(self, num_patches, d_model):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        return x + self.pos_emb  # (B, num_patches, d_model)


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block with multi-head self-attention and FFN."""

    def __init__(self, d_model=128, num_heads=2, ffn_dim=256, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + attn_output

        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = residual + ffn_output
        return x


class RegressionHead(nn.Module):
    """Global average pooling followed by a linear projection to output_dim."""

    def __init__(self, d_model=128, output_dim=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        # x: (B, num_patches, d_model)
        x = x.mean(dim=1)  # Global Average Pooling
        return self.head(x).squeeze(-1)  # (B,)


class BaselineTransformer(nn.Module):
    """Patch-based transformer baseline for PPG → HR estimation."""

    def __init__(self, config, input_dim=None):
        super().__init__()
        input_dim = input_dim if input_dim is not None else getattr(config, "input_dim", None)  # [DUAL-RATE-DESIGN]
        if input_dim is None:
            raise ValueError("BaselineTransformer requires an input_dim; it is derived in BaseTrainer via utils/preprocessing_config.py.")
        patch_size = getattr(config, "patch_size", 25)
        if input_dim % patch_size != 0:
            raise ValueError(
                f"[DUAL-RATE-DESIGN] Derived input_dim={input_dim} is not divisible by patch_size={patch_size}; "
                "this would break the transformer patch embedding."
            )
        d_model = getattr(config, "d_model", 128)
        num_heads = getattr(config, "num_heads", 2)
        ffn_dim = getattr(config, "ffn_dim", 256)
        num_layers = getattr(config, "num_layers", 2)
        output_dim = getattr(config, "output_dim", 1)

        self.patch_embedding = PatchEmbedding(input_dim, patch_size, d_model)
        self.positional_encoding = PositionalEncoding(self.patch_embedding.num_patches, d_model)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        self.regression_head = RegressionHead(d_model, output_dim)

    def forward(self, x):
        # x: (B, input_dim)
        x = self.patch_embedding(x)       # (B, num_patches, d_model)
        x = self.positional_encoding(x)   # (B, num_patches, d_model)

        for encoder in self.encoder_blocks:
            x = encoder(x)               # (B, num_patches, d_model)

        return self.regression_head(x)    # (B,)