"""Bounded-memory assembly using the original BrainPro channel/window operations."""
import json
import os
import pickle
import tempfile
from pathlib import Path
import h5py
import numpy as np
from channels import (
    predefined_channels,
    channel_mapping,
    build_channel_metadata,
    group_channels_by_name_and_parity,
    channel_mask,
    align_eeg,
)
import torch
from torch.utils.data import Dataset

CATEGORIES = ('AFFCT', 'MOTOR', 'OTHER')
WINDOW = 2000

def load_config(path, input_dir):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    raw = json.loads(Path(path).read_text(), object_pairs_hook=unique_keys)
    if not isinstance(raw, dict) or set(raw) != set(CATEGORIES):
        raise ValueError('Config must contain AFFCT, MOTOR and OTHER lists (empty lists allowed).')
    categories, seen = {}, set()
    for category in CATEGORIES:
        files = raw[category]
        if not isinstance(files, list) or not all(isinstance(f, str) and f for f in files):
            raise ValueError(f'{category} must be a list of filenames.')
        categories[category] = files
        for filename in files:
            relative = Path(filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Use paths relative to --input-dir: {filename}')
            source = (Path(input_dir) / relative).resolve()
            if source in seen:
                raise ValueError(f'Duplicate corpus in config: {filename}')
            seen.add(source)
            if not source.is_file():
                raise FileNotFoundError(source)
    if not seen:
        raise ValueError('The config contains no corpus files.')
    return categories, seen

def make_metadata(categories):
    names, numbers, _ = build_channel_metadata(predefined_channels)
    return dict(predefined_channels=predefined_channels, channel_names=names,
        number_indices=numbers,
        group_list=group_channels_by_name_and_parity(names, numbers, predefined_channels),
        dataset_category=categories,
        file_to_label={f: i for i, cat in enumerate(CATEGORIES) for f in categories[cat]},
        affect_important_channel=channel_mask(predefined_channels, ['F','T','C']),
        motor_important_channel=channel_mask(predefined_channels, ['C','P'], ['FP1','FPZ','FP2']))

def canonical_channels(raw_names):
    names = [(x.decode('utf-8') if isinstance(x, bytes) else str(x)).strip().upper() for x in raw_names]
    reverse = {alias.upper(): target.upper() for target, aliases in channel_mapping.items() for alias in aliases}
    selected = {}
    for i, name in enumerate(names):
        canonical = reverse.get(name, name)
        if canonical not in predefined_channels:
            continue
        if canonical not in selected or name == canonical:
            selected[canonical] = i
    if not selected:
        raise ValueError('Recording has no recognized BrainPro channels.')
    return list(selected), list(selected.values())

def build_dataset(config, input_dir, output, metadata_path, stride=2000):
    if stride < 1:
        raise ValueError('Stride must be positive.')
    categories, inputs = load_config(config, input_dir)
    output, metadata_path = Path(output).resolve(), Path(metadata_path).resolve()
    if output == metadata_path or output in inputs or metadata_path in inputs:
        raise ValueError('Output, metadata and input paths must be distinct.')
    for path in [output, metadata_path]:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
    metadata = make_metadata(categories)
    temporary = []
    for path in [output, metadata_path]:
        fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.partial', dir=path.parent)
        os.close(fd)
        temporary.append(Path(name))
    report = dict(windows=0, rejected_windows=0, short_recordings=0, corpora=[])
    try:
        with h5py.File(temporary[0], 'w') as target:
            eeg_out = target.create_dataset('eeg', shape=(0,60,WINDOW+2), maxshape=(None,60,WINDOW+2),
                dtype=np.float32, chunks=(32,60,WINDOW+2))
            labels_out = target.create_dataset('labels', shape=(0,2), maxshape=(None,2), dtype=np.int32, chunks=(32,2))
            target.attrs['chOrder'] = predefined_channels
            target.attrs['sampling_rate_hz'] = 200
            target.attrs['window_size'] = WINDOW
            target.attrs['stride_size'] = stride
            target.attrs['dataset_category_json'] = json.dumps(categories)
            buffer, labels = [], []
            def flush():
                if not buffer:
                    return
                start = eeg_out.shape[0]
                end = start + len(buffer)
                eeg_out.resize(end, axis=0)
                labels_out.resize(end, axis=0)
                eeg_out[start:end] = np.stack(buffer)
                labels_out[start:end] = np.asarray(labels, dtype=np.int32)
                buffer.clear(); labels.clear()
            dataset_id = 0
            for process_id, category in enumerate(CATEGORIES):
                for filename in categories[category]:
                    montage = None
                    count = 0
                    with h5py.File(Path(input_dir)/filename, 'r') as source:
                        for recording in source.keys():
                            group = source[recording]
                            if not isinstance(group, h5py.Group) or 'eeg' not in group:
                                raise ValueError(f'{filename}/{recording}: expected a group containing eeg.')
                            eeg = group['eeg']
                            if not isinstance(eeg,h5py.Dataset) or eeg.ndim != 2 or 'chOrder' not in eeg.attrs:
                                raise ValueError(f'{filename}/{recording}: expected eeg[C,T] with chOrder attribute.')
                            raw_names = eeg.attrs['chOrder']
                            if np.asarray(raw_names).ndim != 1 or len(raw_names) != eeg.shape[0]:
                                raise ValueError(f'{filename}/{recording}: chOrder does not match channel count.')
                            names, indices = canonical_channels(raw_names)
                            current = frozenset(names)
                            if montage is not None and current != montage:
                                raise ValueError(f'{filename}: recordings have different mapped channel sets. '
                                    'Split each montage into its own corpus file/config entry before training.')
                            montage = current
                            if eeg.shape[1] < WINDOW:
                                report['short_recordings'] += 1
                            for start in range(0, eeg.shape[1]-WINDOW+1, stride):
                                segment = np.asarray(eeg[:,start:start+WINDOW],dtype=np.float32)[indices]
                                segment = align_eeg(segment,names,predefined_channels,metadata['group_list'],pad=True)
                                if not np.all(np.abs(segment) <= 1000):
                                    report['rejected_windows'] += 1
                                    continue
                                buffer.append(segment)
                                labels.append((dataset_id,process_id))
                                count += 1
                                report['windows'] += 1
                                if len(buffer) == 32:
                                    flush()
                    if not count:
                        raise ValueError(f'{filename}: no usable 10-second windows.')
                    report['corpora'].append(dict(file=filename,dataset_id=dataset_id,process_id=process_id,windows=count))
                    dataset_id += 1
            flush()
        with temporary[1].open('wb') as handle:
            pickle.dump(metadata,handle)
        # Each destination was checked above; only newly created output files are published.
        if output.exists() or metadata_path.exists():
            raise FileExistsError('An output appeared while building; refusing to overwrite it.')
        temporary[0].rename(output)
        temporary[1].rename(metadata_path)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return report

class PreprocessedEEGDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None  # will be opened lazily in __getitem__

        # Read metadata only once
        with h5py.File(h5_path, 'r') as f:
            self.length = f['eeg'].shape[0]
            self.ch_names = f.attrs['chOrder']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            # Each worker opens its own handle
            self.file = h5py.File(self.h5_path, 'r')

        eeg = self.file['eeg'][idx]
        label = self.file['labels'][idx]

        # Convert to torch tensors
        eeg = torch.from_numpy(eeg).float()
        label = torch.tensor(label).long()

        return eeg, label

    def get_ch_names(self):
        return self.ch_names

    def __del__(self):
        if self.file is not None:
            self.file.close()
