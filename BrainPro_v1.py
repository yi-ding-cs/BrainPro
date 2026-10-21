"""BrainPro-base v1: standalone downstream fine-tuning and inference.

Dependencies: torch, numpy, einops. No timm, registry, or project imports.
Input: float tensor [batch, channels, time] at 200 Hz, with rows in
channel_names order. Apply the dataset's preprocessing before calling this
model. The release loaders divide their processed EEG by 100; this module
does not normalize inputs automatically.

Example:
    model = BrainPro(channel_names=CHANNELS['BCI-IV-2A'], n_times=800,
                     n_classes=4, branch=1, mode='avg', hidden_factor=1,
                     dropout=0.2, checkpoint='brainpro-base.pth')
    logits = model(eeg)

Binary tasks use n_classes=1 and BCEWithLogitsLoss; multiclass tasks use
CrossEntropyLoss. Outputs are logits, without sigmoid or softmax.

Original model attribution:
LaBraM https://github.com/935963004/LaBraM/tree/main
BEiT-v2 https://github.com/microsoft/unilm/tree/master/beitv2
DeiT https://github.com/facebookresearch/deit/
DINO https://github.com/facebookresearch/dino
BrainPro code: CBCR License 1.0; retain upstream notices.
"""
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

__all__ = ['BrainPro', 'CHANNELS', 'predefined_channels']

CHANNELS = {'FACED': ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'CZ', 'C3', 'C4', 'T7', 'T8', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2'], 'SEED-V': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'], 'BCI-IV-2A': ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'PZ', 'P2', 'POZ'], 'SHU': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'Oz', 'O1', 'O2'], 'MentalArithmetic': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'FZ', 'CZ', 'PZ', 'A1'], 'ATTEN': ['FP1', 'AF5', 'AFZ', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'CZ', 'CP5', 'CP1', 'P7', 'P3', 'PZ', 'POZ', 'O1', 'FP2', 'AF6', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4', 'P8', 'O2'], 'SEED-VII': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'], 'Speech': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8'], 'Mumtaz2016': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']}

class Encoder(nn.Module):
    def __init__(self, num_T=16, num_S=16, input_T=10, fs=200, total_chan=60, total_region=24, patch_size=20,
                 patch_step=20, token_dim=128, emb_dropout=0., dim_atten=32, depth=24, heads=32, dim_mlp=128, **kwargs):
        super().__init__()
        self.temporal_encoder = TemporalConv(out_chans=num_T)
        dim = num_T * input_T * fs
        self.spatial_encoder = SpatialLearner(C=total_chan, R=total_region, K_coarse=num_S, K_fine=num_S, dim=dim)

        ############# Transformer ##############
        # we set input sequence is 10s
        num_patches = int((input_T * fs - patch_size) / patch_step) + 1
        token_dim = int(2 * num_T * num_S * patch_size) if token_dim == 0 else token_dim
        self.to_token = Patcher(patch_size=(int(2 * num_T * num_S), patch_size), stride=(1, patch_step),
                                out_dim=token_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, token_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.pos_embedding)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim=token_dim, depth=depth, heads=heads, dim_head=dim_atten, mlp_dim=dim_mlp,
                                       dropout=0.)
        self.x_CNN = None

    def forward(self, eeg, channel_idx, area_idx):
        # eeg: (B, C_eeg, L); channel_idx & area_idx: (C_pre,)
        x = self.temporal_encoder(eeg)  # -> (B, num_T, C_eeg, L)
        x = self.spatial_encoder(x, channel_idx, area_idx)  # -> (B, 2*num_S*num_T, L)
        # get tokens
        x = torch.unsqueeze(x, dim=1)  # -> (B, 1, 2*num_S*num_T, L)
        x = self.to_token(x)  # -> (B, n, token_dim)
        self.x_CNN = x
        n = x.size(1)
        pos = self.pos_embedding[:, :n]
        x += pos
        x = self.dropout(x)
        x = self.transformer(x)
        return x

    def reset_pos(self):
        nn.init.xavier_uniform_(self.pos_embedding)

    def forward_features(self, eeg, channel_idx, area_idx):
        # eeg: (B, C_eeg, L); channel_idx & area_idx: (C_pre,)
        x = self.temporal_encoder(eeg)  # -> (B, num_T, C_eeg, L)
        x = self.spatial_encoder(x, channel_idx, area_idx)  # -> (B, 2*num_S*num_T, L)
        # get tokens
        x = torch.unsqueeze(x, dim=1)  # -> (B, 1, 2*num_S*num_T, L)
        x = self.to_token(x)  # -> (B, n, token_dim)
        return x

    def forward_context(self, x):
        # x: (B, n, token_dim)
        n = x.size(1)
        pos = self.pos_embedding[:, :n]
        x += pos
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determining the output dimension
        '''
        super().__init__()
        pad = self.get_padding(15)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 1), padding=pad)
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        # x: B, C, T
        B, C, T = x.size()
        x = x.unsqueeze(1)  # -> (B, 1, C, T)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))  # -> (B, K, C, T)
        return x

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))


class SpatialLearner(nn.Module):
    """
    Extract coarse-to-fine spatial information from EEG
    """

    def __init__(self, C, R, K_fine, K_coarse, dim):
        super().__init__()
        self.fine_grade_spatial_filters = nn.Parameter(torch.FloatTensor(C, K_fine), requires_grad=True)
        nn.init.xavier_uniform_(self.fine_grade_spatial_filters)
        self.coarse_grade_spatial_filters = nn.Parameter(torch.FloatTensor(R, K_coarse), requires_grad=True)
        nn.init.xavier_uniform_(self.coarse_grade_spatial_filters)
        self.activation_fine = nn.Sequential(
            nn.GELU(),
            nn.GroupNorm(4, K_fine)
        )
        self.activation_coarse = nn.Sequential(
            nn.GELU(),
            nn.GroupNorm(4, K_coarse)
        )

    def coarse_grade_spatial_learning(self, features, region_idx):
        """
            features: (B, N, D) tensor of batch x N samples x D features
            region_idx: (N,)  tensor of integer region labels (any integers)
            returns:   (B, R, D) tensor of region means,
                       in the order of unique region labels
        """
        # remove the "Z" channels
        idx_keep = region_idx != 0
        features = features[:, idx_keep]
        region_idx = region_idx[idx_keep]

        # 1) remap arbitrary labels to [1..R]
        unique_region, inv = torch.unique(region_idx, return_inverse=True)
        #   unique_region.shape == (R,), inv.shape == (N,) with values in [1..R]

        # 2) make N×R one-hot
        one_hot = F.one_hot(inv, num_classes=unique_region.size(0)).to(features.dtype)  # -> (N, R)

        # 3) sum features into each region: (R, C) x (B, C, D) -> (B, R, D)
        sums = torch.matmul(one_hot.t(), features)

        # 4) counts per region: (N,R).sum(dim=0) -> (R,)
        counts = one_hot.sum(dim=0).unsqueeze(1)  # (R,1)

        # 5) means: (R,D)
        means = sums / counts

        # retrieval coarse-grade spatial kernels and do filtering: (K, R) X (B, R, D) -> (B, K, D)
        output = self.activation_coarse(torch.matmul(self.coarse_grade_spatial_filters[unique_region - 1].T, means))

        return output

    def fine_grade_spatial_learning(self, features, channel_idx):
        # retrieval fine-grade spatial kernels and do filtering: (K, C) X (B, C, D) -> (B, K, D)
        output = self.activation_fine(torch.matmul(self.fine_grade_spatial_filters[channel_idx - 1].T, features))
        return output

    def forward(self, x, channel_idx, region_idx, **kwargs):
        # x: B, K, C, T
        B, K, C, T = x.size()
        x = rearrange(x, 'B K C T -> B C (K T)')
        x_coarse = self.coarse_grade_spatial_learning(
            features=x, region_idx=region_idx
        )
        x_coarse = rearrange(x_coarse, 'B S (K T) -> B (S K) T', K=K, T=T)
        x_fine = self.fine_grade_spatial_learning(
            features=x, channel_idx=channel_idx
        )
        x_fine = rearrange(x_fine, 'B S (K T) -> B (S K) T', K=K, T=T)
        x = torch.cat((x_coarse, x_fine), dim=1)  # -> (B, 2*S*K, T)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Patcher(nn.Module):
    """Full-height, non-overlapping temporal patches."""
    def __init__(self, patch_size, stride, in_chan=1, out_dim=0):
        super().__init__()

        self.patch_height, self.patch_width = patch_size

        if tuple(stride) != (1, self.patch_width):
            raise ValueError(
                "Only non-overlapping temporal patches are supported: "
                "stride must be (1, patch_size[1])."
            )
        self.emd_size = (
            in_chan * self.patch_height * self.patch_width
        )
        if out_dim != 0:
            self.to_out = nn.Sequential(
                nn.Linear(self.emd_size, out_dim),
                nn.GELU(),
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        # x: [batch, input_channels, spatial_features, time]
        b, channels, height, length = x.shape
        width = self.patch_width

        if height != self.patch_height:
            raise ValueError(
                f"Expected spatial height {self.patch_height}, got {height}."
            )
        if length % width != 0:
            raise ValueError(
                f"Time length {length} must be divisible by {width}."
            )
        if channels * height * width != self.emd_size:
            raise ValueError("Input channel count does not match in_chan.")

        patches = x.reshape(b, channels, height, length // width, width)
        patches = patches.permute(0, 3, 1, 2, 4).reshape(
            b, length // width, self.emd_size
        )

        return self.to_out(patches)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BrainPro_classifier(nn.Module):
    def __init__(self, encoders, seq_length, token_dim, channel_names, cognition_type,
                 number_class, linear_prob=True, dp=0.25, T=4, mode='mean',**kwargs):
        super().__init__()
        # we set one general encoder, and 4 cognition encoders
        assert len(encoders) == 4
        self.channel_names_downstream = channel_names
        channel_idx, area_idx = self.get_channel_region_idx()
        self.register_buffer('channel_idx', channel_idx)
        self.register_buffer('area_idx', area_idx)
        self.y_cognition = cognition_type
        self.encoders = nn.ModuleList(encoders)  # list of 5 encoder models
        self.mode = mode
        if mode == 'all':
            pass
        elif mode == 'avg':
            seq_length = seq_length // 5
        else:
            seq_length = 1
        self.classifier = nn.Linear(int(2*seq_length*token_dim), number_class) if linear_prob else nn.Sequential(
            nn.Linear(int(2*seq_length*token_dim), T * 200),
            nn.ELU(),
            nn.Dropout(dp),
            nn.Linear(T * 200, 200),
            nn.ELU(),
            nn.Dropout(dp),
            nn.Linear(200, number_class)
        )

    def reset_pos(self):
        for i in range(len(self.encoders)):
            self.encoders[i].reset_pos()
        print('Positional embeddings are reset!!!')

    def forward(self, eeg):
        """
        eeg: input to encoder (shared input), shape (B, N, L)
        """
        b, c, t = eeg.size()
        if t > 2000:
            eeg, n = pad_to_multiple_2000(eeg)
        else:
            n = 0
        eeg, chan_idx, ar_idx = self.select_useful_data(eeg=eeg, channel_idx=self.channel_idx, area_idx=self.area_idx)
        z_shared = self.encoders[-1](eeg, chan_idx, ar_idx)
        encoder = self.encoders[self.y_cognition]
        z_cognition = encoder(eeg, chan_idx, ar_idx)
        z = torch.cat((z_shared, z_cognition), dim=-1)
        if self.mode == 'mean':
            z = torch.mean(z, dim=1)
        elif self.mode == 'avg':
            B, N, L = z.size()
            # reshape into groups of 5
            x_grouped = z.reshape(B, N // 5, 5, L)  # (B, N/5, 5, L)
            # average across the 5
            z = x_grouped.mean(dim=2)  # (B, N/5, L)
            z = rearrange(z, 'B N L -> B (N L)')
        elif self.mode == 'all':
            z = rearrange(z, 'B N L -> B (N L)')

        z = self.classifier(z)
        if n > 0:
            z = rearrange(z, '(B n) O -> B n O', n=n)
            z = torch.mean(z, dim=1)
        return z

    def get_channel_region_idx(self):
        # return the channel_idx and region_idx
        channel_names, number_indices, _ = build_channel_metadata(predefined_channels)
        group_list = group_channels_by_name_and_parity(channel_names, number_indices, predefined_channels)
        channel_idx, area_idx = generate_channel_group_index(channel_names=self.channel_names_downstream,
                                                             predefined_channels=predefined_channels,
                                                             group_list=group_list)
        channel_idx = torch.tensor(channel_idx, dtype=torch.long)
        area_idx = torch.tensor(area_idx, dtype=torch.long)

        return channel_idx, area_idx

    def select_useful_data(self, eeg, channel_idx, area_idx):
        # select the useful data for downstream tasks
        valid_channels = channel_idx != 0  # shape: (C,)
        used_eeg = eeg[:, valid_channels, :]  # shape: (batch, C_valid, T)
        used_channel_idx = channel_idx[valid_channels]
        used_area_idx = area_idx[valid_channels]
        return used_eeg, used_channel_idx, used_area_idx


def pad_to_multiple_2000(x):
    """
    Pads the last dimension of x (shape: b, c, t) with zeros so that t becomes the next multiple of 2000.
    Returns the padded tensor and n (the number of 1024 blocks).
    """
    b, c, t = x.shape
    n = math.ceil(t / 2000)
    t_new = n * 2000
    pad_len = t_new - t
    if pad_len == 0:
        x = rearrange(x, "b c (n l) -> (b n) c l", n=n)
        return x, n
    pad = (0, pad_len)
    x_padded = torch.nn.functional.pad(x, pad)
    x_padded = rearrange(x_padded, "b c (n l) -> (b n) c l", n=n)
    return x_padded, n


predefined_channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
                       'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
                       'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2',
                       'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5',
                       'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']


def parse_channel_number(ch):
    match = re.match(r'([A-Za-z]+)([0-9]+|z)', ch)
    if match:
        return match.groups()
    return ch.upper(), 'z'


def build_channel_metadata(predefined_channels):
    channel_names = []
    number_indices = []
    channel_ids = []
    channel_to_index = {ch.upper(): i for i, ch in enumerate(predefined_channels)}

    for ch in predefined_channels:
        ch = ch.upper()
        name, num = parse_channel_number(ch)
        channel_names.append(name)
        number_indices.append(int(num) if num.isdigit() else -1)
        channel_ids.append(channel_to_index[ch])

    return channel_names, number_indices, channel_ids


def group_channels_by_name_and_parity(channel_names, number_indices, predefined_channels):
    grouped = defaultdict(list)
    for idx, (name, num) in enumerate(zip(channel_names, number_indices)):
        if num == -1:  # skip 'Z' channels for grouping
            continue
        elif num % 2 == 0:
            grouped[(name, 'R')].append(idx)
        else:
            grouped[(name, 'L')].append(idx)

    ordered_groups = []
    seen = set()
    for ch in predefined_channels:
        name, num = parse_channel_number(ch.upper())
        if num.lower() == 'z':
            continue  # skip Z channels
        for gtype in ['L', 'R']:
            key = (name, gtype)
            if key in grouped and key not in seen:
                ordered_groups.append((f"{name}-{gtype}", grouped[key]))
                seen.add(key)

    return ordered_groups


def generate_channel_group_index(channel_names, predefined_channels, group_list):
    """
    Args:
        channel_names: List[str] — current EEG channel names
        predefined_channels: List[str] — full 64-channel list
        group_list: List[Tuple[str, List[int]]] — group name + indices

    Returns:
        channel_idx: numpy.array of size (C,), area_idx (C,), they start with 1 to be consistent with pre-training
    """
    channel_idx_map = {ch: i for i, ch in enumerate(predefined_channels)}

    # Build lookup from channel index to group index
    channel_to_group_idx = {}
    for gid, (_, ch_indices) in enumerate(group_list):
        for ch_idx in ch_indices:
            channel_to_group_idx[ch_idx] = gid

    C = len(predefined_channels)

    ch_idxs, group_idxs = [], []
    for i, ch in enumerate(channel_names):
        ch = ch.upper()
        if ch not in channel_idx_map:
            ch_idx, group_idx = -1, -1
            print("The channel {} is not in pre-defined channel list".format(ch))
        else:
            ch_idx = channel_idx_map.get(ch, -1)
            group_idx = channel_to_group_idx.get(ch_idx, -1)
            name, num = parse_channel_number(ch)
            if num.lower() == 'z':
                group_idx = -1  # set region index to 0 later
        ch_idxs.append(ch_idx)
        group_idxs.append(group_idx)

    group_idxs = np.array(group_idxs)
    ch_idxs = np.array(ch_idxs)
    ch_idxs += 1  # +1 to keep consistency with the padded version
    group_idxs += 1
    assert len(ch_idxs) == len(channel_names), "Mismatched channel index"
    assert len(group_idxs) == len(channel_names), "Mismatched region index"
    return (ch_idxs, group_idxs)


class BrainPro(BrainPro_classifier):
    """Direct BrainPro-base classifier; no model registry is required.

    branch: 0=affective, 1=motor, 2=other (always combined with shared).
    n_times: samples per window, a multiple of 20, at most 2000.
    mode: 'all', 'avg' (groups of five tokens), or 'mean'.
    checkpoint_kind: 'pretrained' loads encoder weights and resets temporal
        position embeddings only when freeze_encoders=False.
        'finetuned' strictly restores the entire classifier, including its
         learned positional embeddings, without resetting them.
    linear_head: use the release's linear-probe head rather than its MLP.
    freeze_encoders: disable encoder gradients, independently of head choice.
    """

    def __init__(self, *, channel_names, n_times, n_classes, branch=0,
                 mode='mean', hidden_factor=1, dropout=0.1, checkpoint=None,
                 checkpoint_kind='pretrained', linear_head=False,
                 freeze_encoders=False):
        if not isinstance(n_times, int) or not 20 <= n_times <= 2000 or n_times % 20:
            raise ValueError('n_times must be a multiple of 20 between 20 and 2000.')
        if not isinstance(n_classes, int) or n_classes < 1:
            raise ValueError('n_classes must be positive; use 1 for binary tasks.')
        if branch not in (0, 1, 2):
            raise ValueError('branch must be 0, 1, or 2.')
        if mode not in ('mean', 'avg', 'all'):
            raise ValueError('mode must be mean, avg, or all.')
        if mode == 'avg' and (n_times // 20) % 5:
            raise ValueError('avg mode requires a multiple of five temporal tokens.')
        if not isinstance(hidden_factor, int) or hidden_factor < 1:
            raise ValueError('hidden_factor must be a positive integer.')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1).')
        if checkpoint_kind not in ('pretrained', 'finetuned'):
            raise ValueError('checkpoint_kind must be pretrained or finetuned.')
        names = list(channel_names)
        if not names or any(not isinstance(name, str) for name in names):
            raise ValueError('channel_names must be a nonempty sequence of strings.')
        recognized = [name.upper() for name in names if name.upper() in predefined_channels]
        if not recognized:
            raise ValueError('No supplied channels are recognized by BrainPro.')
        if all(name.endswith('Z') for name in recognized):
            raise ValueError('At least one recognized non-midline channel is required.')
        encoders = [Encoder(num_T=32, num_S=32, fs=200, total_chan=60,
                            total_region=24, patch_size=20, patch_step=20,
                            token_dim=64, emb_dropout=dropout, dim_atten=32,
                            depth=4, heads=32, dim_mlp=64) for _ in range(4)]
        super().__init__(encoders=encoders, seq_length=n_times // 20,
                         token_dim=64, channel_names=names, cognition_type=branch,
                         number_class=n_classes, linear_prob=linear_head,
                         dp=dropout, T=hidden_factor, mode=mode)
        self.n_times = n_times
        self.n_input_channels = len(names)
        state = self._read_checkpoint(checkpoint) if checkpoint is not None else None
        if state is not None and checkpoint_kind == 'pretrained':
            encoder_state = {k[len('encoders.'):]: v for k, v in state.items()
                             if k.startswith('encoders.')}
            if not encoder_state:
                raise ValueError('Checkpoint has no encoders.* tensors.')
            self.encoders.load_state_dict(encoder_state, strict=True)
        # Preserve the release's initialization order and positional reset.
        if not freeze_encoders and checkpoint_kind != "finetuned":
            self.reset_pos()
        if state is not None and checkpoint_kind == 'finetuned':
            self.load_state_dict(state, strict=True)
        if freeze_encoders:
            self.encoders.requires_grad_(False)

    @staticmethod
    def _read_checkpoint(checkpoint):
        if isinstance(checkpoint, (str, Path)):
            # Load only trusted files; compatible with the release's PyTorch 2.1.
            checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError('checkpoint must be a path or a checkpoint mapping.')
        state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        if not isinstance(state, Mapping) or not state:
            raise ValueError('Checkpoint does not contain a nonempty state_dict.')
        if all(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        return state

    def forward(self, eeg):
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (self.n_input_channels, self.n_times):
            raise ValueError(f'Expected [batch, {self.n_input_channels}, {self.n_times}], '
                             f'got {tuple(eeg.shape)}.')
        return super().forward(eeg)


if __name__=="__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = dict(
        channel_names=["F3", "F4", "C3", "C4", "P3", "P4"],
        n_times=800,  # 4 seconds at 200 Hz
        n_classes=4,
        branch=1,  # 0: affective, 1: motor, 2: other
        mode="all",
        hidden_factor=1,
        dropout=0.1,
    )

    model = BrainPro(
        **model_config,
        checkpoint="/home/dingyi/brainpro/checkpoints/brainpro-base.pth",
        checkpoint_kind="pretrained",
    ).to(device)

    # Synthetic input only to demonstrate the interface; replace with processed EEG.
    eeg = torch.randn(2, 6, 800, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(eeg)  # [2, 4]; no softmax is applied