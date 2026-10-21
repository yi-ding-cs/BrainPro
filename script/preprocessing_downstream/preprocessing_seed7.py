from pathlib import Path
import mne
import glob
import scipy.io as sio
import pdb
import pickle
import numpy as np
import os
import sys
import lmdb

raw_data_path = os.environ["BRAINPRO_RAW_DIR"]
preprocessed_data_path = os.environ["BRAINPRO_OUTPUT_DIR"]
os.makedirs(preprocessed_data_path, exist_ok=True)

rawDataPath = Path(raw_data_path)
group = rawDataPath.glob('*.mat')
group = [path for path in group if 'label' not in path.name]
files = sorted(group, key=lambda x: x.name)
print(files)

video_order = (
    ["happy","neutral","disgust","sad","anger","anger","sad","disgust","neutral","happy"] * 2 +
    ["anger","sad","fear","neutral","surprise","surprise","neutral","fear","sad","anger"] * 2 +
    ["happy","surprise","disgust","fear","anger","anger","fear","disgust","surprise","happy"] * 2 +
    ["disgust","sad","fear","surprise","happy","happy","surprise","fear","sad","disgust"] * 2
)
label_dict = {
    "happy": 0,
    "surprise": 1,
    "neutral": 2,
    "sad": 3,
    "disgust": 4,
    "fear": 5,
    "anger": 6,
}
labels = [ label_dict[v] for v in video_order ]

segment_length = 1

trials_split = {
    'train': range(10),
    'val': range(10, 15),
    'test': range(15, 20),
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db = lmdb.open(preprocessed_data_path, map_size=205614542346)

for file in files:
    print(file)
    data_trials = sio.loadmat(file)
    items = [(int(k.split('_')[1]), data_trials[k]) for k in data_trials.keys() if k.startswith('eeg_')]
    items.sort(key=lambda t: t[0])

    session_len = 20  # trials per session

    for sess_id in range(len(items) // session_len):
        sess_start = sess_id * session_len
        sess_items = items[sess_start:sess_start + session_len]

        for mode, idx_range in trials_split.items():
            for idx_in_sess in idx_range:
                idx_trial, data = sess_items[idx_in_sess]
                label = labels[idx_trial - 1]

                print('{}: idx_trial{} label{}'.format(mode, idx_trial, label))

                # segment
                C, T = data.shape
                block = segment_length * 200
                T_trim = (T // block) * block
                arr_trim = data[:, :T_trim]

                data_segments = arr_trim.reshape(62, -1, segment_length, 200)
                data_segments = data_segments.transpose(1, 0, 2, 3)  # (n_segments, 62, seg_len, 200)

                print(f"Session {sess_id + 1}, trial {idx_trial}, shape={data_segments.shape}")

                for i, sample in enumerate(data_segments):
                    sample_key = f'{file.stem}-sess{sess_id + 1}-trial{idx_trial}-seg{i}'
                    data_dict = {'sample': sample.astype(np.float32), 'label': label}

                    with db.begin(write=True) as txn:
                        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))

                    dataset[mode].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()

