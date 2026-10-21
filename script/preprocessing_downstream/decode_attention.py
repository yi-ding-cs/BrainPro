import os
import numpy as np
import os.path as osp
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)


def load_original_data(path):
    # this function load the .eeg files
    raw = mne.io.read_raw_brainvision(path)
    raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog'})
    mne.rename_channels(raw.info, {'FP1': 'Fp1', 'FP2': 'Fp2'})

    montage = mne.channels.make_standard_montage(kind='standard_1005')
    raw.set_montage(montage)
    return raw

def ICA_EOG_removal(raw, plot=False, filtering=True):
    # This function use EOG channels and ICA to remove EOGs
    raw.load_data()

    eog_evoked = create_eog_epochs(raw, ch_name=['HEOG', 'VEOG']).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))

    filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
    ica = ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(filt_raw)

    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices

    if plot:
        raw.plot()
        eog_evoked.plot()
        # barplot of ICA component "EOG match" scores
        ica.plot_scores(eog_scores)

        # plot ICs applied to raw data, with EOG matches highlighted
        ica.plot_sources(raw, show_scrollbars=False)

        # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        ica.plot_sources(eog_evoked)
        ica.plot_sources(raw, show_scrollbars=False)
        # blinks
        ica.plot_overlay(raw, exclude=[0], picks='eeg')

    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    if plot:
        raw.plot()
        reconst_raw.plot()

    if filtering:
        reconst_raw = reconst_raw.filter(.5, None, method='iir')
        reconst_raw = reconst_raw.filter(None, 50., method='iir')

    return reconst_raw

def get_epochs(raw):
    # This function helps to cut the continuous data into epochs (trials)
    events = mne.events_from_annotations(raw)
    events = events[0]

    # get event information of attention task
    events_attention = events.copy()

    # get event information of rest task
    events_rest = events.copy()
    idx_first_trial = np.where(events_rest[:, -1] == 48)[0][
        0]  # 48 is the start cule of one block (40s attention + 20s rest)
    events_rest[idx_first_trial][-1] = 0  # disable the start index of the first trial
    events_rest = np.concatenate((events_rest, np.array([[events_rest[-1][0] + 20 * 1000, 0, 48]])),
                                 axis=0)  # add the ending index of the last trial
    # get epochs according to the event codes
    '''
    Here we use the first half of the attention task to balance the data samples between attention 
    and rest as:
    Zhang, Yangsong, et al. "An end-to-end 3D convolutional neural network for decoding attentive mental state." 
    Neural Networks 144 (2021): 129-137.
    '''
    epochs_attention = mne.Epochs(raw, events_attention, event_id=[48], tmin=0.0, tmax=20.0,
                                  baseline=None)
    epochs_rest = mne.Epochs(raw, events_rest, event_id=[48], tmin=-20.0, tmax=0.0,
                             baseline=None)
    # load the data and do resampling
    epochs_attention.load_data()
    epochs_rest.load_data()
    epochs_attention = epochs_attention.resample(sfreq=200)
    epochs_rest = epochs_rest.resample(sfreq=200)

    # get the data
    data_attention = epochs_attention.get_data()
    data_rest = epochs_rest.get_data()
    return data_attention, data_rest


num_ses=3
num_subject=26
path=os.environ["BRAINPRO_RAW_DIR"]
name_dataset='ATTEN'
save_path = os.environ["BRAINPRO_OUTPUT_DIR"]
# loop each session
for ses in range(1, num_ses + 1):
    # loop each subject
    for sub in range(1, num_subject + 1):
        if sub <= 9:
            sub_code = '0{}'.format(sub)
        else:
            sub_code = sub

        path_load = os.path.join(path, 'VP0{}'.format(sub_code), 'gonogo{}.vhdr'.format(ses))
        raw = load_original_data(path_load)
        # plot_eeg(raw.get_data()[:, :10000])
        reconst_raw = ICA_EOG_removal(raw=raw, plot=False, filtering=True)
        # plot_eeg(reconst_raw.get_data()[:, :10000])
        # plot_eeg(data[-1])

        reconst_raw.pick_types(eeg=True)

        data_att, data_rest = get_epochs(raw=reconst_raw)

        data_att = data_att * 1e6
        data_rest = data_rest * 1e6

        data = np.concatenate((data_att, data_rest), axis=0)
        label = np.array([1, 0]).repeat(data_att.shape[0])
        # plot_eeg(data[0])
        # plot_eeg(data[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        np.save(osp.join(save_path, 'sub{}_ses{}_label'.format(sub, ses)), label)
        np.save(osp.join(save_path, 'sub{}_ses{}_eeg'.format(sub, ses)), data)
        print('Sub{} Ses{} is saved'.format(sub, ses))