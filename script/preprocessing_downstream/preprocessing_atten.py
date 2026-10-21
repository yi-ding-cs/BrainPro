import os
import numpy as np
import pickle
import lmdb
from scipy import signal

# ========== Config ==========
root_dir = os.environ["BRAINPRO_RAW_DIR"]  # where sub*_ses*_eeg.npy lives
preprocessed_data_path = os.environ["BRAINPRO_OUTPUT_DIR"]
os.makedirs(preprocessed_data_path, exist_ok=True)

subjects = list(range(1, 27))   # 26 subjects
sessions = [1, 2, 3]
segment_length = 4  # seconds
fs = 200           # sampling rate (Hz)
seg_samples = segment_length * fs
# ============================

# LMDB storage
db = lmdb.open(preprocessed_data_path, map_size=20 * 1024**3)  # 20 GB map size
dataset = {'train': [], 'val': [], 'test': []}

# simple split: 80% train, 10% val, 10% test (by subject)

splits = {
    'train': subjects[:20],
    'val': subjects[20:23],
    'test': subjects[23:]
}

for split_name, sub_list in splits.items():
    for sub in sub_list:
        data_one_sub = []
        label_one_sub = []
        for ses in sessions:
            eeg_path = os.path.join(root_dir, f'sub{sub}_ses{ses}_eeg.npy')
            label_path = os.path.join(root_dir, f'sub{sub}_ses{ses}_label.npy')
            if not (os.path.exists(eeg_path) and os.path.exists(label_path)):
                continue

            eeg = np.load(eeg_path, allow_pickle=True)   # shape: (trials, ch, time)
            label = np.load(label_path, allow_pickle=True)  # shape: (trials,)
            data_one_sub.append(eeg)
            label_one_sub.append(label)
        data_one_sub = np.concatenate(data_one_sub, axis=0)
        label_one_sub = np.concatenate(label_one_sub, axis=0)
        data_one_sub_ = data_one_sub.reshape(36, 28, 20, 200)

        for i, (samples, label) in enumerate(zip(data_one_sub_, label_one_sub)):
            for j in range(samples.shape[1] // segment_length):
                sample = samples[:, segment_length*j:segment_length*(j+1), :]
                sample_key = f'sub{sub}-{i}-{j}'
                print(sample_key)
                data_dict = {
                    'sample': sample, 'label': label
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[split_name].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
