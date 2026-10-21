# --------------------------------------------------------
# Towards Large-Scale Cognition-aware EEG Representation Learning
# By Ding Yi
# Based on LaBraM, BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/935963004/LaBraM/tree/main
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.registry import register_model
from timm.models import create_model
import math
import numpy as np


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


class Decoder(nn.Module):
    def __init__(self, num_T=16, num_S=16, input_T=10, fs=200, total_chan=60, total_region=24, patch_size=20,
                 patch_step=20, token_dim=128, emb_dropout=0., dim_atten=32, depth=24, heads=32, dim_mlp=128, **kwargs):
        super().__init__()
        ############# Transformer ##############
        # we set maximum input sequence is 10s
        num_patches = int((input_T * fs - patch_size) / patch_step) + 1
        token_dim = int(2 * num_T * num_S * patch_size) if token_dim == 0 else token_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, token_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.pos_embedding)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim=token_dim, depth=depth, heads=heads, dim_head=dim_atten, mlp_dim=dim_mlp,
                                       dropout=0.)
        self.to_out = nn.Sequential(
            nn.Linear(token_dim, int(total_chan * patch_size)),  # match the target size
            Rearrange('B N (C P) -> B C (N P)', C=total_chan, P=patch_size)
        )

    def forward(self, x, match_size=False):
        """
        x: (B, n, token_dim)
        """
        n = x.size(1)
        pos = self.pos_embedding[:, :n]
        x += pos
        x = self.dropout(x)
        x = self.transformer(x)
        if match_size:
            x = self.to_out(x)  # (B, Total_chan, n*patch_size)
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
    def __init__(self, patch_size, stride, in_chan=1, out_dim=0):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.emd_size = int(in_chan * patch_size[0] * patch_size[1])
        if out_dim != 0:
            self.to_out = nn.Sequential(
                nn.Linear(int(in_chan * patch_size[0] * patch_size[1]), out_dim),
                nn.GELU()
            )
        else:
            self.to_out = nn.Identity()
        self.to_patch = Rearrange("b l n -> b n l")

    def forward(self, x):
        # x: b, k, c, l
        # Fast path for full-height, non-overlapping temporal patches.
        b, channels, height, length = x.shape
        kh, kw = self.unfold.kernel_size
        sh, sw = self.unfold.stride

        if (kh == height and sh == 1 and sw == kw
                and length % kw == 0
                and self.unfold.padding == 0
                and self.unfold.dilation == 1):
            x = x.reshape(b, channels, height, length // kw, kw)
            x = x.permute(0, 3, 1, 2, 4).reshape(
                b, length // kw, channels * height * kw)
        else:
            x = self.to_patch(self.unfold(x))

        return self.to_out(x)  # -> b, n, l

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


class BrainProPreTrain(nn.Module):
    def __init__(self, encoders, decoders, embed_dim, **kwargs):
        super().__init__()
        # we set one general encoder, and 3 cognition encoders
        assert len(encoders) == 4 and len(decoders) == 3
        self.encoders = nn.ModuleList(encoders)  # list of 4 encoder models
        self.decoders = nn.ModuleList(decoders)  # list of 3 decoder models
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def patchify_on_eeg(self, x, patch_size):
        """
        x: [N, C, T]
        Returns: [N, C * n_patch, patch_size]
        """
        N, C, T = x.shape
        assert T % patch_size == 0, "T must be divisible by patch_size"
        n_patch = T // patch_size
        # Split time into patches, then combine C and n_patch
        x_patch = rearrange(x, 'n c (np p) -> n (c np) p', np=n_patch, p=patch_size)
        return x_patch

    def unpatchify_on_eeg(self, x_patch, C, T):
        """
        x_patch: [N, C * n_patch, patch_size]
        Returns: [N, C, T]
        """
        N, cnp, patch_size = x_patch.shape
        n_patch = T // patch_size
        # Separate combined axis back to C and n_patch, then join time
        x = rearrange(x_patch, 'n (c np) p -> n c (np p)', c=C, np=n_patch, p=patch_size)
        return x

    def apply_maks(self, x, bool_masked_pos):
        # x: (B, C*N_patch, T/N_patch)
        batch_size, seq_len, _ = x.size()
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # replace the masked tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        return x

    def forward(self, eeg, channel_idx, area_idx, y_cognition, bool_masked_pos=None, patch_size=0):
        """
        eeg: input to encoder (shared input), shape (B, N, L)
        channel_idx: used for retrieval for fine-grade spatial kernels (N,)
        area_idx: used for retrieval for coarse-grade spatial kernels (N,)
        y_cognition: which encoder to activate
        bool_masked_pos: masked positions (bool)
        patch_size: the length of the masked patch

        Returns:
            z_shared: representation from shared encoder, (B, N, token_dim)
            z_cognition: representation from the activated encoder, (B, N, token_dim)
            z_other_types: representation from other encoders, [(B, N, token_dim), ...]
            z_TS: representation from temporal-spatial block, (B, N, token_dim)
            z_rec: representation from decoder, (B, N, token_dim)
        """
        if bool_masked_pos is None:
            input_eeg = eeg
        else:
            # mask eeg
            B, C, T = eeg.size()
            masked_eeg = self.patchify_on_eeg(eeg, patch_size)
            masked_eeg = self.apply_maks(masked_eeg, bool_masked_pos)
            input_eeg = self.unpatchify_on_eeg(masked_eeg, C, T)

        # get the latent embedding
        z_shared = self.encoders[-1](input_eeg, channel_idx, area_idx)
        z_cognition = None
        z_other_types = []
        z_comb = None
        for j in range(0, 3):
            encoder = self.encoders[j]
            if j != y_cognition:
                z_j = encoder(input_eeg, channel_idx, area_idx)
                z_j = z_j.detach()  # only update y_cognition's encoder
                z_other_types.append(z_j)
            else:
                z_j = encoder(input_eeg, channel_idx, area_idx)  # used for decoupling loss
                z_cognition = z_j  # used for decoupling loss
                z_comb = torch.cat((z_j, z_shared), dim=-1)

        # get the reconstructed
        z_rec = self.decoders[y_cognition](z_comb, match_size=True)  # for reconstruction loss
        z_rec = z_rec[:, channel_idx-1]  # select the corresponding channels
        return z_shared, z_cognition, z_other_types, eeg, z_rec


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@register_model
def brainpro_base(pretrained=False, **kwargs):  # 23.61M
    encoders = [Encoder(num_T=32, num_S=32, fs=200, total_chan=60, total_region=24, patch_size=20, patch_step=20,
                        token_dim=64, emb_dropout=0., dim_atten=32, depth=4, heads=32, dim_mlp=64) for _ in range(4)]
    decoders = [Decoder(num_T=32, num_S=32, fs=200, total_chan=60, total_region=24, patch_size=20, patch_step=20,
                        token_dim=128, emb_dropout=0., dim_atten=32, depth=2, heads=32, dim_mlp=64) for _ in range(3)]
    model = BrainProPreTrain(encoders=encoders, decoders=decoders, embed_dim=20)
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model



# for downstream usage
@register_model
def BrainPro(pretrained=False, **kwargs):  # 18.08M
    encoders = [Encoder(num_T=32, num_S=32, fs=200, total_chan=60, total_region=24, patch_size=20, patch_step=20,
                        token_dim=64, emb_dropout=kwargs["dropout"], dim_atten=32, depth=4, heads=32, dim_mlp=64) for _ in range(4)]
    model = BrainPro_classifier(
        encoders=encoders, seq_length=kwargs["seq_length"], token_dim=64, channel_names=kwargs["channel_names"],
        cognition_type=kwargs["cognition_type"], number_class=kwargs["number_class"],
        linear_prob=kwargs["linear_prob"], dp=kwargs["dropout"], T=kwargs["T"], mode=kwargs["mode"]
    )
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        pretrained_state = checkpoint["model"] if "model" in checkpoint else checkpoint
        encoder_state = {k.replace("encoders.", ""): v
                         for k, v in pretrained_state.items()
                         if k.startswith("encoders.")}
        missing, unexpected = model.encoders.load_state_dict(encoder_state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    if not kwargs["params"].frozen:
        model.reset_pos()

    return model


def decouple_cosine_margin_loss(z1, z2, margin=0.1):
    sim = F.cosine_similarity(z1, z2, dim=-1)  # (B, N)
    if margin != 0:
        loss = torch.clamp(sim - margin, min=0).mean()
    else:
        loss = sim.mean()
    return loss


def brain_region_aware_reconstruction_loss(channel_importance, z_rec, z_target, T):
    m = channel_importance
    # Compute weights smoothly:
    weights = 0.5 + torch.sigmoid(T * (m - 0.5))

    # Repeat the mask to an entire sample
    _, C, T = z_target.size()
    weights = weights.unsqueeze(1).expand(C, T)

    loss = torch.mean(weights * (z_target - z_rec) ** 2)
    return loss


def loss_fun_enhanced(channel_importance, z_shared, z_cognition, z_other_types,
                      z_target, z_rec, T=1, Lambda1=1.0, Lambda2=1.0):
    # === Decoupling loss ===
    loss_decouple = 0
    # Shared vs each type-specific
    loss_decouple += decouple_cosine_margin_loss(z_shared, z_cognition)
    # Active type-specific vs others
    for z_j in z_other_types:
        loss_decouple += decouple_cosine_margin_loss(z_cognition, z_j)

    # === Reconstruction ===
    loss_recon = brain_region_aware_reconstruction_loss(channel_importance, z_rec, z_target, T)

    # === Final loss ===
    loss = Lambda1 * loss_recon + Lambda2 * loss_decouple
    return loss, loss_recon, loss_decouple


# === Predefined 60-channel configs ===
predefined_channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
                       'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
                       'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2',
                       'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5',
                       'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
# use this to match channels
channel_mapping = {
    "Fp1": ['Fp1', 'AF7', 'Fp1h', 'AFp3', 'AFp5h', 'AFp5', 'AFp7h', 'AF7h', 'AFp7', 'AFp9h', 'AFp9'],
    "Fpz": ['Fpz', 'AFp1h', 'AFp2h', 'AFp1', 'AFpz', 'AFp2'],
    "Fp2": ['Fp2', 'AF8', 'Fp2h', 'AFp4', 'AFp6h', 'AFp6', 'AFp8h', 'AF8h', 'AFp8', 'AFp10h', 'AFp10'],
    "AF3": ['AF1', 'AF1h', 'AF3h', 'AF3', 'AFF3h', 'AF5', 'AF5h', 'AFz'],
    "AF4": ['AF2', 'AF2h', 'AF4h', 'AF4', 'AFF4h', 'AF6', 'AF6h'],
    "F7": ['F7', 'FFT7h', 'AFF7', 'AF9', 'F9', 'AFF9h', 'AF9h', 'F9h', 'AFF9'],
    "F5": ['F5', 'FFC5h', 'AFF5', 'FFC5', 'AFF7h', 'F7h'],
    "F3": ['F3', 'AFF3', 'FFC3', 'F5h'],
    "F1": ['F1', 'AFF1', 'FFC3h'],
    "Fz": ['Fz', 'AFF1h', 'FFC1h', 'F1h', 'AFFz'],
    "F2": ['F2', 'AFF2', 'FFC4h'],
    "F4": ['F4', 'AFF4', 'FFC4', 'F6h'],
    "F6": ['F6', 'FFC6h', 'AFF6', 'FFC6', 'AFF8h', 'F8h'],
    "F8": ['F8', 'FFT8h', 'AFF8', 'AF10', 'F10', 'AFF10h', 'AF10h', 'F10h', 'AFF10'],
    "FT7": ['FT7', 'FFT7', 'FTT7h', 'FT9', 'FFT9h', 'FFT9', 'FT9h', 'FTT9'],
    "FC5": ['FC5', 'FC5h', 'FT7h'],
    "FC3": ['FC3', 'FC3h'],
    "FC1": ['FC1', 'FCC1', 'FC1h'],
    "FCz": ['FCz', 'FFCz', 'FCCz'],
    "FC2": ['FC2', 'FCC2'],
    "FC4": ['FC4', 'FC4h'],
    "FC6": ['FC6', 'FC6h', 'FT8h'],
    "FT8": ['FT8', 'FFT8', 'FTT8h', 'FT10', 'FFT10h', 'FFT10', 'FT10h', 'FTT10'],
    "T7": ['T7', 'A1', 'T3', 'T9', 'T9h', 'TTP9'],
    "C5": ['C5', 'FCC5h', 'FCC5', 'CCP5'],
    "C3": ['C3', 'CCP3', 'C5h'],
    "C1": ['C1', 'CCP1', 'C3h'],
    "Cz": ['Cz', 'CCPz'],
    "C2": ['C2', 'CCP2', 'C4h'],
    "C4": ['C4', 'CCP4', 'C6h'],
    "C6": ['C6', 'FCC6h', 'FCC6', 'CCP6'],
    "T8": ['T8', 'A2', 'T4', 'T10', 'T10h', 'TTP10'],
    "TP7": ['TP9', 'TP7', 'TTP7h', 'TPP9h', 'TP9h', 'TP7h', 'TTP7', 'TPP9', 'M1'],
    "CP5": ['CP5', 'CCP5h'],
    "CP3": ['CP3', 'CCP3h'],
    "CP1": ['CP1', 'CP1h'],
    "CPz": ['CPz'],
    "CP2": ['CP2', 'CP2h'],
    "CP4": ['CP4', 'CCP4h'],
    "CP6": ['CP6', 'CCP6h'],
    "TP8": ['TP8', 'TP10', 'TTP8h', 'TTP10h', 'TPP10h', 'TP10h', 'TPP8', 'TPP10', 'M2'],
    "P7": ['P7', 'T5', 'TPP7h', 'P9', 'PPO9h', 'P9h', 'PPO9'],
    "P5": ['P5', 'CPP5h'],
    "P3": ['P3', 'P3h', 'CPP3', 'PPO3', 'P5h'],
    "P1": ['P1', 'CPP1', 'PPO1', 'CPP1h', 'P1h'],
    "Pz": ['Pz', 'CPPz'],
    "P2": ['P2', 'CPP2', 'PPO2', 'CPP2h', 'P2h'],
    "P4": ['P4', 'P4h', 'CPP4', 'PPO4', 'P6h'],
    "P6": ['P6', 'CPP6h'],
    "P8": ['P8', 'T6', 'TPP8h', 'P10', 'PPO10h', 'P10h', 'PPO10'],
    "PO7": ['PO7', 'PPO7', 'PO9', 'PPO7h', 'POO9h', 'PO9h', 'POO9'],
    "PO5": ['PO5', 'PPO5h', 'PO5h'],
    "PO3": ['PO3', 'PPO3h', 'PO3h', 'POO3'],
    "POz": ['POz', 'PPOz'],
    "PO4": ['PO4', 'POO4h', 'PO4h', 'POO4'],
    "PO6": ['PO6', 'PPO6h', 'PO6h'],
    "PO8": ['PO8', 'PPO8', 'PO10', 'PPO8h', 'POO10h', 'PO10h', 'POO10'],
    "O1": ['O1', 'OI1h', 'O1h', 'I1h', 'I1', 'OI1', 'POO5h', 'POO5', 'POO7h', 'POO7'],
    "Oz": ['Oz', 'Iz', 'POO1h', 'POO2h', 'POO1', 'POOz', 'POO2', 'OIz'],
    "O2": ['O2', 'OI2h', 'O2h', 'I2h', 'I2', 'OI2', 'POO6h', 'POO6', 'POO8h', 'POO8'],
}

# === Parse channel names like AF7 → ('AF', 7)
def parse_channel_number(ch):
    match = re.match(r'([A-Za-z]+)([0-9]+|z)', ch)
    if match:
        return match.groups()
    return ch.upper(), 'z'

# === Build channel metadata
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

# === Build groupings by name and odd/even index (excluding 'Z' channels)
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


