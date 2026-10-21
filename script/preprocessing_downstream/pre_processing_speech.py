import h5py
import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def fit_norm(X):
    bounds = np.percentile(X, [0.1,99.9]).astype(X.dtype)
    clipped = np.clip(X,*bounds)
    q25,median,q75 = np.percentile(clipped,[25,50,75]).astype(X.dtype)
    if not np.isfinite(q75-q25) or q75 <= q25:
        raise ValueError('Training normalization has invalid/zero IQR.')
    return bounds,median,q75-q25

def norm(X,stats):
    bounds,median,iqr=stats
    return (np.clip(X,*bounds)-median)/iqr

normalizers = {}

train_dir = os.path.join(os.environ["BRAINPRO_RAW_DIR"], "Training set")
test_dir = os.path.join(os.environ["BRAINPRO_RAW_DIR"], "Validation set")
#test_dir = '/data/datasets/BigDownstream/Imagined speech/mat/Test set'
preprocessed_data_path = os.environ["BRAINPRO_OUTPUT_DIR"]
os.makedirs(preprocessed_data_path, exist_ok=True)

files_dict = {
    'train':sorted([file for file in os.listdir(train_dir)]),
    'test':sorted([file for file in os.listdir(test_dir)]),
}

print(files_dict)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db = lmdb.open(preprocessed_data_path, map_size=3000000000)

for file in files_dict['train']:
    data = scipy.io.loadmat(os.path.join(train_dir, file))
    print(data['epo_train'][0][0][0])
    eeg = data['epo_train'][0][0][4].transpose(2, 1, 0)
    labels = data['epo_train'][0][0][5].transpose(1, 0)

    eeg_padded = np.pad(
        eeg,
        pad_width=((0, 0), (0, 0), (0, 5)),  # pad only the last axis (time)
        mode='constant',
        constant_values=0
    )
    #eeg = eeg[:, :, -768:]
    labels = np.argmax(labels, axis=1)
    eeg = eeg_padded.reshape(300, 64, 4, 200)
    #eeg = signal.resample(eeg, 600, axis=2).reshape(300, 64, 3, 200)
    print(eeg.shape, labels.shape)

    train_idx, val_idx = train_test_split(
        np.arange(eeg.shape[0]), test_size=0.1, stratify=labels, random_state=42
    )

    # Fit only on original (unpadded) training samples, then transform held-out trials.
    stats = fit_norm(eeg.reshape(300,64,800)[train_idx,:,:795])
    normalizers[str(data['epo_train']['title'][0,0].item())] = stats
    unpadded = norm(eeg.reshape(300,64,800)[:,:,:795],stats)
    eeg = np.pad(unpadded,((0,0),(0,0),(0,5))).reshape(300,64,4,200)
    data_train = eeg[train_idx]
    label_train = labels[train_idx]
    for i, (sample, label) in enumerate(zip(data_train, label_train)):
        sample_key = f'train-{file[:-4]}-{i}'
        data_dict = {
            'sample': sample, 'label': label,
        }
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
        txn.commit()
        print(sample_key)
        dataset['train'].append(sample_key)

    data_val = eeg[val_idx]  # (channels, time)
    label_val = labels[val_idx]
    for i, (sample, label) in enumerate(zip(data_val, label_val)):
        sample_key = f'val-{file[:-4]}-{i}'
        data_dict = {
            'sample': sample, 'label': label,
        }
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
        txn.commit()
        print(sample_key)
        dataset['val'].append(sample_key)


for file in files_dict['test']:
    data = scipy.io.loadmat(os.path.join(test_dir, file))
    eeg = data['epo_validation'][0][0][4].transpose(2, 1, 0)
    labels = data['epo_validation'][0][0][5].transpose(1, 0)
    subject = str(data['epo_validation']['title'][0,0].item())
    if subject not in normalizers:
        raise ValueError(f'No training normalization for {subject}')
    eeg = norm(eeg,normalizers[subject])

    eeg_padded = np.pad(
        eeg,
        pad_width=((0, 0), (0, 0), (0, 5)),  # pad only the last axis (time)
        mode='constant',
        constant_values=0
    )
    labels = np.argmax(labels, axis=1)
    eeg = eeg_padded.reshape(50, 64, 4, 200)
    #eeg = eeg[:, :, -768:]
    #labels = np.argmax(labels, axis=1)
    #eeg = signal.resample(eeg, 600, axis=2).reshape(50, 64, 3, 200)
    print(eeg.shape, labels.shape)
    for i, (sample, label) in enumerate(zip(eeg, labels)):
        sample_key = f'test-{file[:-4]}-{i}'
        data_dict = {
            'sample': sample, 'label': label,
        }
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
        txn.commit()
        print(sample_key)
        dataset['test'].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()





