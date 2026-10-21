import scipy
from scipy import signal
import os
import lmdb
import pickle
import mne

root_dir = os.environ["BRAINPRO_RAW_DIR"]
preprocessed_data_path = os.environ["BRAINPRO_OUTPUT_DIR"]
os.makedirs(preprocessed_data_path, exist_ok=True)
files = [file for file in os.listdir(root_dir) if file.endswith('.edf')]
files = sorted(files)
print(files)
segment_length = 4 # in seconds

files_dict = {
    'train':files[:56],
    'val':files[56:64],
    'test':files[64:],
}
print(files_dict)
dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}


selected_channels = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4',
                     'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2',
                     'EEG Fz', 'EEG Cz', 'EEG Pz', 'EEG A2-A1']


db = lmdb.open(preprocessed_data_path, map_size=1000000000)
for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        raw = mne.io.read_raw_edf(os.path.join(root_dir, file), preload=True)
        raw.pick(selected_channels)
        raw.reorder_channels(selected_channels)
        raw.filter(l_freq=0.5, h_freq=45)
        raw.resample(200)

        eeg = raw.get_data(units='uV')
        chs, points = eeg.shape
        a = points % (segment_length * 200)
        if a != 0:
            eeg = eeg[:, :-a]
        eeg = eeg.reshape(20, -1, segment_length, 200).transpose(1, 0, 2, 3)
        label = int(file[-5])
        if label == 1:
            eeg = eeg[:15]

        for i, sample in enumerate(eeg):
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