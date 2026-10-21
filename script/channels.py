import re
from collections import defaultdict
import numpy as np

# Canonical channel order and aliases used by BrainPro-base.
predefined_channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
                       'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                       'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                       'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']

channel_mapping = {'Fp1': ['Fp1', 'AF7', 'Fp1h', 'AFp3', 'AFp5h', 'AFp5', 'AFp7h', 'AF7h', 'AFp7', 'AFp9h', 'AFp9'],
 'Fpz': ['Fpz', 'AFp1h', 'AFp2h', 'AFp1', 'AFpz', 'AFp2'],
 'Fp2': ['Fp2', 'AF8', 'Fp2h', 'AFp4', 'AFp6h', 'AFp6', 'AFp8h', 'AF8h', 'AFp8', 'AFp10h', 'AFp10'],
 'AF3': ['AF1', 'AF1h', 'AF3h', 'AF3', 'AFF3h', 'AF5', 'AF5h', 'AFz'],
 'AF4': ['AF2', 'AF2h', 'AF4h', 'AF4', 'AFF4h', 'AF6', 'AF6h'],
 'F7': ['F7', 'FFT7h', 'AFF7', 'AF9', 'F9', 'AFF9h', 'AF9h', 'F9h', 'AFF9'],
 'F5': ['F5', 'FFC5h', 'AFF5', 'FFC5', 'AFF7h', 'F7h'],
 'F3': ['F3', 'AFF3', 'FFC3', 'F5h'],
 'F1': ['F1', 'AFF1', 'FFC3h'],
 'Fz': ['Fz', 'AFF1h', 'FFC1h', 'F1h', 'AFFz'],
 'F2': ['F2', 'AFF2', 'FFC4h'],
 'F4': ['F4', 'AFF4', 'FFC4', 'F6h'],
 'F6': ['F6', 'FFC6h', 'AFF6', 'FFC6', 'AFF8h', 'F8h'],
 'F8': ['F8', 'FFT8h', 'AFF8', 'AF10', 'F10', 'AFF10h', 'AF10h', 'F10h', 'AFF10'],
 'FT7': ['FT7', 'FFT7', 'FTT7h', 'FT9', 'FFT9h', 'FFT9', 'FT9h', 'FTT9'],
 'FC5': ['FC5', 'FC5h', 'FT7h'],
 'FC3': ['FC3', 'FC3h'],
 'FC1': ['FC1', 'FCC1', 'FC1h'],
 'FCz': ['FCz', 'FFCz', 'FCCz'],
 'FC2': ['FC2', 'FCC2'],
 'FC4': ['FC4', 'FC4h'],
 'FC6': ['FC6', 'FC6h', 'FT8h'],
 'FT8': ['FT8', 'FFT8', 'FTT8h', 'FT10', 'FFT10h', 'FFT10', 'FT10h', 'FTT10'],
 'T7': ['T7', 'A1', 'T3', 'T9', 'T9h', 'TTP9'],
 'C5': ['C5', 'FCC5h', 'FCC5', 'CCP5'],
 'C3': ['C3', 'CCP3', 'C5h'],
 'C1': ['C1', 'CCP1', 'C3h'],
 'Cz': ['Cz', 'CCPz'],
 'C2': ['C2', 'CCP2', 'C4h'],
 'C4': ['C4', 'CCP4', 'C6h'],
 'C6': ['C6', 'FCC6h', 'FCC6', 'CCP6'],
 'T8': ['T8', 'A2', 'T4', 'T10', 'T10h', 'TTP10'],
 'TP7': ['TP9', 'TP7', 'TTP7h', 'TPP9h', 'TP9h', 'TP7h', 'TTP7', 'TPP9', 'M1'],
 'CP5': ['CP5', 'CCP5h'],
 'CP3': ['CP3', 'CCP3h'],
 'CP1': ['CP1', 'CP1h'],
 'CPz': ['CPz'],
 'CP2': ['CP2', 'CP2h'],
 'CP4': ['CP4', 'CCP4h'],
 'CP6': ['CP6', 'CCP6h'],
 'TP8': ['TP8', 'TP10', 'TTP8h', 'TTP10h', 'TPP10h', 'TP10h', 'TPP8', 'TPP10', 'M2'],
 'P7': ['P7', 'T5', 'TPP7h', 'P9', 'PPO9h', 'P9h', 'PPO9'],
 'P5': ['P5', 'CPP5h'],
 'P3': ['P3', 'P3h', 'CPP3', 'PPO3', 'P5h'],
 'P1': ['P1', 'CPP1', 'PPO1', 'CPP1h', 'P1h'],
 'Pz': ['Pz', 'CPPz'],
 'P2': ['P2', 'CPP2', 'PPO2', 'CPP2h', 'P2h'],
 'P4': ['P4', 'P4h', 'CPP4', 'PPO4', 'P6h'],
 'P6': ['P6', 'CPP6h'],
 'P8': ['P8', 'T6', 'TPP8h', 'P10', 'PPO10h', 'P10h', 'PPO10'],
 'PO7': ['PO7', 'PPO7', 'PO9', 'PPO7h', 'POO9h', 'PO9h', 'POO9'],
 'PO5': ['PO5', 'PPO5h', 'PO5h'],
 'PO3': ['PO3', 'PPO3h', 'PO3h', 'POO3'],
 'POz': ['POz', 'PPOz'],
 'PO4': ['PO4', 'POO4h', 'PO4h', 'POO4'],
 'PO6': ['PO6', 'PPO6h', 'PO6h'],
 'PO8': ['PO8', 'PPO8', 'PO10', 'PPO8h', 'POO10h', 'PO10h', 'POO10'],
 'O1': ['O1', 'OI1h', 'O1h', 'I1h', 'I1', 'OI1', 'POO5h', 'POO5', 'POO7h', 'POO7'],
 'Oz': ['Oz', 'Iz', 'POO1h', 'POO2h', 'POO1', 'POOz', 'POO2', 'OIz'],
 'O2': ['O2', 'OI2h', 'O2h', 'I2h', 'I2', 'OI2', 'POO6h', 'POO6', 'POO8h', 'POO8']}

channels_down_stream_dataset = {
    'FACED': ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'CZ', 'C3', 'C4', 'T7', 'T8', 'A1',
              'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2'],
    'SEED-V': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
               'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
               'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
               'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'],
    'BCI-IV-2A': ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CP1',
                  'CPZ', 'CP2', 'CP4', 'P1', 'PZ', 'P2', 'POZ'],
    'SHU': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T3', 'T4', 'A1',
            'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'Oz', 'O1', 'O2'],
    'MentalArithmetic': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1',
                         'O2', 'FZ', 'CZ', 'PZ', 'A1'],
    'ATTEN': ['FP1', 'AF5', 'AFZ', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'CZ', 'CP5', 'CP1', 'P7', 'P3', 'PZ', 'POZ', 'O1',
              'FP2', 'AF6', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4', 'P8', 'O2'],
    'SEED-VII': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
                 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'],
    'Speech': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8',
               'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10',
               'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1',
               'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8'],
    'Mumtaz2016': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                   'FZ', 'CZ', 'PZ'],
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

def align_eeg(eeg_array, channel_names, predefined_channels, group_list, pad=False):
    """
    Args:
        eeg_array: numpy.array (C_sub, T)
        channel_names: List[str] — current EEG channel names
        predefined_channels: List[str] — full 64-channel list
        group_list: List[Tuple[str, List[int]]] — group name + indices
        pad: whether to do zero padding or not

    Returns:
        Array of shape (C, T + 2) if pad == True else Array of shape (C, 2)
    """
    channel_idx_map = {ch: i for i, ch in enumerate(predefined_channels)}

    # Build lookup from channel index to group index
    channel_to_group_idx = {}
    for gid, (_, ch_indices) in enumerate(group_list):
        for ch_idx in ch_indices:
            channel_to_group_idx[ch_idx] = gid

    C = len(predefined_channels)
    T = eeg_array.shape[-1]
    if pad:
        padded = np.zeros((C, T+2), dtype=eeg_array.dtype)

    ch_idxs, group_idxs = [], []
    for i, ch in enumerate(channel_names):
        ch = ch.upper()
        assert ch in channel_idx_map, "The channel {} is not in the pre-defined channels!".format(ch)
        ch_idx = channel_idx_map.get(ch, -1)
        group_idx = channel_to_group_idx.get(ch_idx, -1)
        name, num = parse_channel_number(ch)
        if num.lower() == 'z':
            group_idx = -1  # set region index to 0 later
        ch_idxs.append(ch_idx)
        group_idxs.append(group_idx)
        if pad:
            padded[ch_idx, :T] = eeg_array[i]
            padded[ch_idx, T] = ch_idx + 1  # +1 to avoid conflict between index 0 and the padded 0s.
            padded[ch_idx, T + 1] = group_idx + 1 if group_idx >= 0 else 0
    if pad:
        return padded
    else:
        group_idxs = np.array(group_idxs)
        ch_idxs = np.array(ch_idxs)
        ch_idxs += 1 # +1 to keep consistency with the padded version
        group_idxs += 1
        chan_info = np.stack((ch_idxs, group_idxs), axis=1)  # (C_sub, 2)
        return chan_info

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

def channel_mask(predefined_list, regions, except_channels=None):
    regions = [l.upper() for l in regions]  # make case-insensitive
    except_channels = set([ch.upper() for ch in (except_channels or [])])  # case-insensitive set
    mask = []
    for channel in predefined_list:
        ch_upper = channel.upper()
        if ch_upper in except_channels:
            mask.append(0)
        else:
            mask.append(int(any(letter in ch_upper for letter in regions)))
    return mask