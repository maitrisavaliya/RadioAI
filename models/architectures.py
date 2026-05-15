"""
Novel architectures for RadioAI — exactly matching trained checkpoints:
  - DPMS_LSW  : CT Scan  (Dual-Path Multi-Scale + Learnable Slice Weighting)
  - TARNet    : Ultrasound (Tissue Acoustic Response Network)
  - MSCAF     : MRI      (Multi-Scale CNN-Attention Fusion)

Every attribute name, layer order, and dimension matches the training notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models as tv_models


# ─────────────────────────────────────────────────────────────────────────────
# CT  ─  DPMS-LSW
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleSpatialAttention(nn.Module):
    """
    Multi-scale spatial attention — 3 kernel sizes fused with learnable weights.
    Attribute names: scale1, scale2, scale3, fusion_weights
    Each scale ends with Sigmoid, producing attention maps multiplied onto x.
    """
    def __init__(self, in_channels):
        super().__init__()
        mid = in_channels // 8

        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, in_channels, 1), nn.Sigmoid()
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, in_channels, 1), nn.Sigmoid()
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, mid, 5, padding=2), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, in_channels, 1), nn.Sigmoid()
        )
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        w = F.softmax(self.fusion_weights, dim=0)
        att = w[0] * self.scale1(x) + w[1] * self.scale2(x) + w[2] * self.scale3(x)
        return x * att


class LearnableSliceWeighting(nn.Module):
    """
    Learns which slices (augmentation views) matter most.
    Attribute names: slice_importance, context_net
    """
    def __init__(self, num_slices, feature_dim):
        super().__init__()
        self.slice_importance = nn.Parameter(torch.ones(1, num_slices, 1))
        self.context_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4), nn.ReLU(),
            nn.Linear(feature_dim // 4, 1), nn.Sigmoid()
        )

    def forward(self, slice_features):
        # slice_features: (batch, num_slices, feature_dim)
        static_weights  = torch.softmax(self.slice_importance, dim=1)
        context_weights = self.context_net(slice_features)          # (B, S, 1)
        combined        = static_weights * context_weights
        combined        = combined / (combined.sum(dim=1, keepdim=True) + 1e-8)
        return slice_features * combined, combined


class CrossPathFusion(nn.Module):
    """
    Fuses local (single-slice) and global (all-slices) features via cross-attention.
    Attribute names: local_proj, global_proj, cross_attention, fusion_gate, output_proj
    """
    def __init__(self, local_dim, global_dim, output_dim):
        super().__init__()
        self.local_proj  = nn.Linear(local_dim,  output_dim)
        self.global_proj = nn.Linear(global_dim, output_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim), nn.Sigmoid()
        )
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim), nn.ReLU(), nn.Dropout(0.2)
        )

    def forward(self, local_features, global_features):
        lp = self.local_proj(local_features).unsqueeze(1)    # (B, 1, D)
        gp = self.global_proj(global_features).unsqueeze(1)  # (B, 1, D)

        attended, _ = self.cross_attention(lp, gp, gp)
        attended     = attended.squeeze(1)                    # (B, D)

        gate   = self.fusion_gate(
            torch.cat([attended, self.global_proj(global_features)], dim=1)
        )
        fused  = gate * attended + (1 - gate) * self.global_proj(global_features)
        output = self.output_proj(torch.cat([fused, attended], dim=1))
        return output


class DPMS_LSW(nn.Module):
    """
    Dual-Path Multi-Scale Attention Network with Learnable Slice Weighting.

    Local Path : EfficientNet-B2 on middle slice + MultiScaleSpatialAttention
    Global Path: ResNet-34 on all slices + LearnableSliceWeighting + Transformer
    Fusion      : CrossPathFusion (cross-attention gated)
    Head        : Standard classifier

    Attribute names exactly match dpms_lsw.pth:
      local_encoder, local_attention, local_pool
      global_encoder, slice_embed, slice_weighting, temporal_encoder
      fusion, classifier
    """
    CLASSES = ["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Normal"]

    def __init__(self, num_classes=4, num_slices=8):
        super().__init__()
        self.num_slices = num_slices

        # ── LOCAL PATH ──────────────────────────────────────────────────────
        self.local_encoder = tv_models.efficientnet_b2(weights=None)
        self.local_encoder.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.local_encoder.classifier = nn.Identity()
        self.local_attention = MultiScaleSpatialAttention(1408)
        self.local_pool      = nn.AdaptiveAvgPool2d(1)

        # ── GLOBAL PATH ─────────────────────────────────────────────────────
        self.global_encoder = tv_models.resnet34(weights=None)
        self.global_encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.global_encoder.fc = nn.Identity()

        self.slice_embed    = nn.Linear(512, 384)
        self.slice_weighting = LearnableSliceWeighting(num_slices, 384)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # ── FUSION ──────────────────────────────────────────────────────────
        self.fusion = CrossPathFusion(local_dim=1408, global_dim=384, output_dim=512)

        # ── CLASSIFIER ──────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, num_slices, 1, H, W)
        batch_size = x.size(0)

        # LOCAL: middle slice
        mid        = self.num_slices // 2
        local_feat = self.local_encoder.features(x[:, mid])   # (B, 1408, H', W')
        local_feat = self.local_attention(local_feat)
        local_feat = self.local_pool(local_feat).view(batch_size, -1)  # (B, 1408)

        # GLOBAL: all slices
        slice_feats = []
        for i in range(self.num_slices):
            sf = self.global_encoder(x[:, i])   # (B, 512)
            sf = self.slice_embed(sf)            # (B, 384)
            slice_feats.append(sf)

        slice_feats = torch.stack(slice_feats, dim=1)          # (B, S, 384)
        weighted, _ = self.slice_weighting(slice_feats)
        temporal    = self.temporal_encoder(weighted)           # (B, S, 384)
        global_feat = temporal.mean(dim=1)                      # (B, 384)

        # FUSION + CLASSIFY
        fused = self.fusion(local_feat, global_feat)            # (B, 512)
        return self.classifier(fused)


# ─────────────────────────────────────────────────────────────────────────────
# ULTRASOUND  ─  TARNet
# ─────────────────────────────────────────────────────────────────────────────

class AttenuationBranch(nn.Module):
    """Depth-decay row attention. Attribute names: log_decay, row_gate, norm"""
    def __init__(self, channels=352):
        super().__init__()
        self.log_decay = nn.Parameter(torch.linspace(0, -2, 16))
        self.row_gate  = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.BatchNorm2d(channels // 8), nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        decay = torch.sigmoid(F.interpolate(
            self.log_decay.view(1, 1, -1), size=H, mode="linear", align_corners=False
        )).view(1, 1, H, 1).expand(B, 1, H, W)
        gate = torch.sigmoid(self.row_gate(x))
        return self.norm(x * decay * gate)


class BackscatterBranch(nn.Module):
    """Multi-granularity speckle modeling. Attribute names: impedance_w, fine, medium, coarse, project"""
    def __init__(self, channels=352):
        super().__init__()
        mid = channels // 8
        self.impedance_w = nn.Parameter(torch.ones(3) / 3)
        self.fine   = nn.Sequential(nn.Conv2d(channels, mid, 3, padding=1),
                                    nn.BatchNorm2d(mid), nn.ReLU())
        self.medium = nn.Sequential(nn.Conv2d(channels, mid, 5, padding=2),
                                    nn.BatchNorm2d(mid), nn.ReLU())
        self.coarse = nn.Sequential(nn.Conv2d(channels, mid, 7, padding=3),
                                    nn.BatchNorm2d(mid), nn.ReLU())
        self.project = nn.Sequential(nn.Conv2d(mid, channels, 1),
                                     nn.BatchNorm2d(channels))

    def forward(self, x):
        w = F.softmax(self.impedance_w, 0)
        fused = w[0]*self.fine(x) + w[1]*self.medium(x) + w[2]*self.coarse(x)
        return x + self.project(fused)


class AcousticShadowBranch(nn.Module):
    """Posterior artifact lateral attention. Attribute names: posterior_anchor, col_gate, norm"""
    def __init__(self, channels=352, n_anchor=16):
        super().__init__()
        self.posterior_anchor = nn.Parameter(torch.linspace(0.1, 0.9, n_anchor))
        self.col_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.BatchNorm2d(channels // 8), nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        mask = torch.sigmoid(F.interpolate(
            self.posterior_anchor.view(1, 1, -1), size=H, mode="linear", align_corners=False
        )).view(1, 1, H, 1).expand(B, 1, H, W)
        col = torch.sigmoid(self.col_gate(x))
        return self.norm(x * mask * col)


class AcousticCompositionGate(nn.Module):
    """Gated fusion of the three acoustic branches. Attribute names: gate, out_proj"""
    def __init__(self, channels=352):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels * 3, channels), nn.ReLU(),
            nn.Linear(channels, 3),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, b1, b2, b3):
        combined = torch.cat([b1.mean([2, 3]), b2.mean([2, 3]), b3.mean([2, 3])], dim=-1)
        w = F.softmax(self.gate(combined), dim=-1)
        fused = (w[:, 0:1, None, None] * b1 +
                 w[:, 1:2, None, None] * b2 +
                 w[:, 2:3, None, None] * b3)
        return self.out_proj(fused), w


class TARNet(nn.Module):
    """
    Tissue Acoustic Response Network for Ultrasound.

    Backbone : EfficientNet-B2 (features_only, stage 4 output = 352ch)
    Branch 1 : AttenuationBranch
    Branch 2 : BackscatterBranch
    Branch 3 : AcousticShadowBranch
    Fusion   : AcousticCompositionGate
    Head     : AdaptiveAvgPool -> Flatten -> Linear(352,512) -> Linear(512,128) -> Linear(128,n)

    Attribute names exactly match tarnet.pth:
      backbone, attn_branch, bsc_branch, shd_branch, acg, pool, classifier
    """
    CLASSES = ["Benign", "Malignant", "Normal"]

    def __init__(self, num_classes=3):
        super().__init__()
        backbone = timm.create_model(
            "efficientnet_b2", pretrained=False, features_only=True, out_indices=(4,)
        )
        old = backbone.conv_stem
        backbone.conv_stem = nn.Conv2d(
            1, old.out_channels, old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False
        )
        self.backbone    = backbone
        feat_ch = 352

        self.attn_branch = AttenuationBranch(feat_ch)
        self.bsc_branch  = BackscatterBranch(feat_ch)
        self.shd_branch  = AcousticShadowBranch(feat_ch)
        self.acg         = AcousticCompositionGate(feat_ch)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feat_ch, 512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 128),     nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_branch_weights=False):
        feat = self.backbone(x)[0]
        att  = self.attn_branch(feat)
        bsc  = self.bsc_branch(feat)
        shd  = self.shd_branch(feat)
        fused, branch_w = self.acg(att, bsc, shd)
        out = self.classifier(self.pool(fused))
        if return_branch_weights:
            return out, branch_w
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MRI  ─  MSCAF
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionModule(nn.Module):
    """
    Cross-Attention between CNN features and Transformer tokens.
    Attribute names: query_proj, key_proj, value_proj, out_proj, scale
    """
    def __init__(self, cnn_channels, transformer_dim):
        super().__init__()
        self.query_proj = nn.Conv2d(cnn_channels, transformer_dim, kernel_size=1)
        self.key_proj   = nn.Linear(transformer_dim, transformer_dim)
        self.value_proj = nn.Linear(transformer_dim, transformer_dim)
        self.out_proj   = nn.Conv2d(transformer_dim, cnn_channels, kernel_size=1)
        self.scale      = transformer_dim ** -0.5

    def forward(self, cnn_feat, transformer_feat):
        B, C, H, W = cnn_feat.shape
        query = self.query_proj(cnn_feat).flatten(2).transpose(1, 2)  # (B, HW, D)
        key   = self.key_proj(transformer_feat)                        # (B, N, D)
        value = self.value_proj(transformer_feat)                      # (B, N, D)
        attn  = F.softmax(torch.matmul(query, key.transpose(-2, -1)) * self.scale, dim=-1)
        out   = torch.matmul(attn, value)                              # (B, HW, D)
        out   = out.transpose(1, 2).reshape(B, -1, H, W)
        return cnn_feat + self.out_proj(out)


class MSCAF(nn.Module):
    """
    Multi-Scale CNN-Attention Fusion for MRI brain tumour classification.

    CNN Branch   : 3 progressively downsampled scales (64 / 128 / 256 ch)
    Trans Branch : Patch embed (16×16) → positional encoding → 6-layer Transformer
    Fusion       : per-scale CrossAttention → GAP → concat with transformer CLS
    Head         : Linear(448+384=832, 512) → Linear(512, n_classes)

    Attribute names exactly match mscaf.pth:
      cnn_scale1, cnn_scale2, cnn_scale3
      patch_embed, pos_embed
      transformer_encoder
      cross_attention_1, cross_attention_2, cross_attention_3
      classifier
    """
    CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

    def __init__(self, num_classes=4, img_size=224, embed_dim=384):
        super().__init__()

        # ── Multi-scale CNN ───────────────────────────────────────────────
        self.cnn_scale1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.cnn_scale2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.cnn_scale3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        # ── Transformer ───────────────────────────────────────────────────
        n_patches = (img_size // 16) ** 2  # 196 for 224×224
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=16, stride=16)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=6)

        # ── Cross-attention (one per CNN scale) ───────────────────────────
        self.cross_attention_1 = CrossAttentionModule(64,  embed_dim)
        self.cross_attention_2 = CrossAttentionModule(128, embed_dim)
        self.cross_attention_3 = CrossAttentionModule(256, embed_dim)

        # ── Classifier ────────────────────────────────────────────────────
        cnn_dim   = 64 + 128 + 256     # 448
        total_dim = cnn_dim + embed_dim # 448 + 384 = 832
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # CNN scales
        s1 = self.cnn_scale1(x)
        s2 = self.cnn_scale2(s1)
        s3 = self.cnn_scale3(s2)

        # Transformer
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = tokens + self.pos_embed
        tokens = self.transformer_encoder(tokens)

        # Cross-attention (each CNN scale attends to transformer tokens)
        s1 = self.cross_attention_1(s1, tokens)
        s2 = self.cross_attention_2(s2, tokens)
        s3 = self.cross_attention_3(s3, tokens)

        # GAP each scale, concatenate with transformer mean token
        gap = lambda t: t.mean([2, 3])
        cnn_feat     = torch.cat([gap(s1), gap(s2), gap(s3)], dim=-1)  # (B, 448)
        trans_global = tokens.mean(dim=1)                                # (B, 384)
        fused        = torch.cat([cnn_feat, trans_global], dim=-1)      # (B, 832)
        return self.classifier(fused)