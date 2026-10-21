import mne
import os
import numpy as np
import json
import scipy.io as sio
import csv
import datetime


def get_montage(montage_file_path: str = 'channel_62_pos.locs') -> mne.channels.DigMontage:
    """
    Return the montage specified by the param montage_file_path
    """
    montage = mne.channels.read_custom_montage(montage_file_path)
    return montage


montage_file_path = os.environ["BRAINPRO_MONTAGE"]
sfreq = 200 # resample frequency
l_freq = 0.1
h_freq = 75
freqs = 50


if __name__ == '__main__':
    data_path = os.environ["BRAINPRO_RAW_DIR"]
    preprocessed_data_path = os.environ["BRAINPRO_OUTPUT_DIR"]
    save_info_path = os.environ["BRAINPRO_TRIGGER_DIR"]
    os.makedirs(preprocessed_data_path, exist_ok=True)
    raw_files = os.listdir(data_path)

    file_dict = {}
    for file in raw_files:
        if file[:-15] not in file_dict.keys():
            file_dict[file[:-15]] = [file]
        else:
            file_dict[file[:-15]].append(file)

    for key, value in file_dict.items():
        EEG_preprocessed = {}
        EEG_features = {}
        for raw_file in value:
            print(raw_file)
            raw_path = os.path.join(data_path, raw_file)
            raw = mne.io.read_raw_cnt(raw_path, eog=['HEO', 'VEO'], ecg=['ECG'])

            # Drop useless channels
            raw.drop_channels(['M1', 'M2', 'ECG', 'HEO', 'VEO'])
            montage = get_montage(montage_file_path)
            raw.set_montage(montage)
            raw.load_data()

            # Preprocessing
            raw.filter(l_freq=l_freq, h_freq=h_freq)
            raw.notch_filter(freqs=freqs)
            #raw.resample(sfreq=sfreq, n_jobs=4)
            raw.resample(sfreq=sfreq)

            # Get triggers for each trial
            trigger, _ = mne.events_from_annotations(raw)

            data, times = raw.get_data(units='uV', return_times=True)

            t = trigger[:, 0]

            # The triggers in cnt file are not accurate for these two files, so we use trigger files instead
            if raw_file == "14_20221015_1.cnt":
                t = []
                start = datetime.datetime.strptime('14:25:34', '%H:%M:%S')
                with open(os.path.join(save_info_path, '14_20221015_1_trigger_info.csv')) as f:
                    trigger = csv.reader(f)
                    for row in trigger:
                        end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                        time_diff = end.timestamp() - start.timestamp()
                        t.append(int(round(time_diff * sfreq)))
            elif raw_file == "9_20221111_3.cnt":
                t = []
                start = datetime.datetime.strptime('14:01:27', '%H:%M:%S')
                with open(os.path.join(save_info_path, '9_20221111_3_trigger_info.csv')) as f:
                    trigger = csv.reader(f)
                    for row in trigger:
                        end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                        time_diff = end.timestamp() - start.timestamp()
                        t.append(int(round(time_diff * sfreq)))

            session_idx = int(raw_file[-5]) - 1
            for i in range(20):
                preprocessed_clip = data[:, t[2 * i]:t[2 * i + 1]]
                EEG_preprocessed[f'eeg_{session_idx * 20 + i + 1}'] = preprocessed_clip

        save_path = os.path.join(preprocessed_data_path, '{}.mat'.format(key))
        sio.savemat(save_path, EEG_preprocessed)
        print("Subject{} is saved at {}".format(key, preprocessed_data_path))

