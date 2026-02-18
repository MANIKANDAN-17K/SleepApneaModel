import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

class PatchTSTConfig:
    # Input
    seq_len     = 3000      # 30 seconds × 100 Hz
    n_vars      = 1         # single ECG channel

    # Patch settings
    patch_len   = 50        # each patch = 0.5 seconds
    stride      = 25        # 50% overlap between patches
    
    # Transformer
    d_model     = 128       # embedding dimension
    n_heads     = 8         # attention heads
    n_layers    = 4         # transformer layers
    d_ff        = 256       # feedforward dimension
    dropout     = 0.1       # dropout rate

    # Classification
    n_classes   = 2         # 0=Normal, 1=Apnea

    @property
    def n_patches(self):
        return (self.seq_len - self.patch_len) // self.stride + 1


# ──────────────────────────────────────────────────────────
# PATCH EMBEDDING
# ──────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits ECG signal into patches and projects to d_model dimension.

    Input  : (batch, seq_len, n_vars)
    Output : (batch, n_patches, d_model)
    """
    def __init__(self, cfg):
        super().__init__()
        self.patch_len = cfg.patch_len
        self.stride    = cfg.stride

        # Linear projection from patch_len → d_model
        self.projection = nn.Linear(cfg.patch_len, cfg.d_model)
        self.norm       = nn.LayerNorm(cfg.d_model)
        self.dropout    = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # x : (batch, seq_len, n_vars)
        # Extract patches using unfold
        # (batch, n_vars, seq_len) → unfold → (batch, n_vars, n_patches, patch_len)
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # (batch, n_vars, n_patches, patch_len) → (batch, n_patches, patch_len)
        # Since n_vars=1, squeeze it
        x = x.squeeze(1)

        # Project patches to d_model
        x = self.projection(x)   # (batch, n_patches, d_model)
        x = self.norm(x)
        x = self.dropout(x)
        return x


# ──────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ──────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for patch positions.

    Input  : (batch, n_patches, d_model)
    Output : (batch, n_patches, d_model)
    """
    def __init__(self, cfg):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, cfg.n_patches, cfg.d_model)
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.pos_embedding
        return self.dropout(x)


# ──────────────────────────────────────────────────────────
# TRANSFORMER ENCODER BLOCK
# ──────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block with:
    - Multi-Head Self Attention
    - Feed Forward Network
    - Layer Normalization
    - Residual Connections
    """
    def __init__(self, cfg):
        super().__init__()

        # Multi-Head Self Attention
        self.attention = nn.MultiheadAttention(
            embed_dim   = cfg.d_model,
            num_heads   = cfg.n_heads,
            dropout     = cfg.dropout,
            batch_first = True
        )

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        # Self Attention + Residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN + Residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ──────────────────────────────────────────────────────────
# PatchTST MODEL
# ──────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    PatchTST model for ECG Apnea Classification.

    Architecture:
        Input ECG (3000 samples)
            ↓
        Patch Embedding (50 samples/patch, 25 stride → 119 patches)
            ↓
        Positional Encoding
            ↓
        Transformer Encoder (4 layers)
            ↓
        Global Average Pooling
            ↓
        Classification Head (2 classes)

    Input  : (batch, seq_len, n_vars) = (batch, 3000, 1)
    Output : (batch, n_classes)       = (batch, 2)
    """
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = PatchTSTConfig()
        self.cfg = cfg

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(cfg)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(cfg)

        # Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(cfg)
            for _ in range(cfg.n_layers)
        ])

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.n_classes)
        )

    def forward(self, x):
        # x : (batch, seq_len, n_vars)

        # Patch Embedding
        x = self.patch_embedding(x)     # (batch, n_patches, d_model)

        # Positional Encoding
        x = self.pos_encoding(x)        # (batch, n_patches, d_model)

        # Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x)                # (batch, n_patches, d_model)

        # Global Average Pooling across patches
        x = x.mean(dim=1)              # (batch, d_model)

        # Classification
        x = self.classifier(x)         # (batch, n_classes)
        return x


# ──────────────────────────────────────────────────────────
# MAIN — Model Summary
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg   = PatchTSTConfig()
    model = PatchTST(cfg)

    # Count parameters
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 55)
    print("  PatchTST Model Summary")
    print("=" * 55)
    print(f"Sequence length  : {cfg.seq_len} samples (30 sec)")
    print(f"Patch length     : {cfg.patch_len} samples (0.5 sec)")
    print(f"Stride           : {cfg.stride} samples")
    print(f"Number of patches: {cfg.n_patches}")
    print(f"d_model          : {cfg.d_model}")
    print(f"Attention heads  : {cfg.n_heads}")
    print(f"Encoder layers   : {cfg.n_layers}")
    print(f"Feedforward dim  : {cfg.d_ff}")
    print(f"Dropout          : {cfg.dropout}")
    print(f"Output classes   : {cfg.n_classes}")
    print("=" * 55)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    dummy_input = torch.randn(32, 3000, 1)   # batch=32
    output      = model(dummy_input)

    print("=" * 55)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("✅ PatchTST Forward Pass Successful!")