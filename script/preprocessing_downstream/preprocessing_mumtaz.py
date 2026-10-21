import os
import mne
import numpy as np
import lmdb
import pickle

#遍历文件夹
def iter_files(rootDir):
    #遍历根目录
    files_H, files_MDD = [], []
    for file in os.listdir(rootDir):
        if 'TASK' not in file and file.endswith('.edf'):
            if 'MDD' in file:
                files_MDD.append(file)
            else:
                files_H.append(file)
    return files_H, files_MDD


selected_channels = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                     'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
                     'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE']
rootDir = os.environ["BRAINPRO_RAW_DIR"]
preprocessed_data_path = os.environ["BRAINPRO_OUTPUT_DIR"]
os.makedirs(preprocessed_data_path, exist_ok=True)

files_H, files_MDD = iter_files(rootDir)
files_H = sorted(files_H)
files_MDD = sorted(files_MDD)
print(files_H)
print(files_MDD)
print(len(files_H), len(files_MDD))


files_dict = {
    'train':[],
    'val':[],
    'test':[],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

# Keep all EC/EO recordings from a subject in one split. Preserve the
# original lexicographic subject ordering and intended subject quotas.
def subject_id(filename):
    import re
    match = re.match(r'(H|MDD) S(\d+) ', filename)
    if not match:
        raise ValueError(f'Unrecognized subject filename: {filename}')
    return match.group(1), match.group(2)

for recordings, ntrain, nval in [(files_H,20,4),(files_MDD,21,5)]:
    subjects = list(dict.fromkeys(subject_id(f) for f in recordings))
    if len(subjects) <= ntrain+nval:
        raise ValueError('Not enough subjects for the configured train/validation/test split.')
    for i, subject in enumerate(subjects):
        split = 'train' if i < ntrain else 'val' if i < ntrain+nval else 'test'
        files_dict[split].extend(f for f in recordings if subject_id(f)==subject)

print(files_dict['train'])
print(files_dict['val'])
print(files_dict['test'])

segment_length = 4  # in seconds

db = lmdb.open(preprocessed_data_path, map_size=2273741824)

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        raw = mne.io.read_raw_edf(os.path.join(rootDir, file), preload=True)
        print(raw.info['ch_names'])
        raw.pick_channels(selected_channels, ordered=True)
        print(raw.info['ch_names'])
        raw.resample(200)
        raw.filter(l_freq=0.3, h_freq=75)
        raw.notch_filter((50))
        # raw.plot_psd(average=True)
        eeg_array = raw.to_data_frame().values
        # print(raw.info)
        eeg_array = eeg_array[:, 1:]
        points, chs = eeg_array.shape
        print(eeg_array.shape)
        a = points % (segment_length * 200)
        print(a)
        if a != 0:
            eeg_array = eeg_array[:-a, :]
        eeg_array = eeg_array.reshape(-1, segment_length, 200, chs)
        eeg_array = eeg_array.transpose(0, 3, 1, 2)
        print(eeg_array.shape)
        label = 1 if 'MDD' in file else 0
        for i, sample in enumerate(eeg_array):
            sample_key = f'{file[:-4]}_{i}'
            data_dict = {
                'sample': sample, 'label': label
            }
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
            txn.commit()
            dataset[files_key].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()