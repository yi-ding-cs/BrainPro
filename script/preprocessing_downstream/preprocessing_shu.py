import scipy
from scipy import signal
import os
import lmdb
import pickle
from pathlib import Path

root_dir = Path(os.environ["BRAINPRO_RAW_DIR"])
# Collect all .mat files under S01-S25 subfolders
mat_files = sorted(root_dir.glob("S*/d*.mat"))
# Convert to list of strings
files = [str(f) for f in mat_files]

# print(files)

files_dict = {
    'train':files[:75],
    'val':files[75:100],
    'test':files[100:],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}
db = lmdb.open(os.environ["BRAINPRO_OUTPUT_DIR"], map_size=2010612736)
for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        data = scipy.io.loadmat(file)
        eeg = data['data']
        labels = data['labels'][0]
        bz, ch_num, points = eeg.shape
        print(eeg.shape)
        eeg_resample = signal.resample(eeg, 800, axis=2)
        eeg_ = eeg_resample.reshape(bz, ch_num, 4, 200)
        print(eeg_.shape, labels.shape)
        for i, (sample, label) in enumerate(zip(eeg_, labels)):
            sample_key = f'{file[:-4]}-{i}'
            # print(sample_key)
            data_dict = {
                'sample':sample, 'label':label-1
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            txn.commit()
            dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()