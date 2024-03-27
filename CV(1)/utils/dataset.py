import os

import numpy as np
import scipy
from scipy import io

import torch
from torch.utils.data import Dataset


class SeismicDataset(Dataset):
    def __init__(self, meta, hdf_data,
                 label_shape='gaussian',
                 label_width=30,
                 wave_length=None,
                 is_train=True):
        self.meta = meta            # csv metadata
        self.data = hdf_data        # hdf5 data
        self.label_shape = label_shape
        assert self.label_shape in ['gaussian', 'triangle', 'box']
        self.label_width = label_width
        self.wave_length = wave_length
        self.is_train = is_train

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        trace_name = self.meta.loc[self.meta.index[idx], 'trace_name']
        waveforms = self.data['data'][trace_name]   # shaped 3 (channels) x wave_length
        waveforms = np.array(waveforms, dtype=float)[:, :self.wave_length]
        waveforms -= np.mean(waveforms, axis=-1, keepdims=True)
        waveforms /= np.amax(waveforms)

        # arrival time of P wave
        tmp_p_t = self.meta.loc[self.meta.index[idx], 'trace_P_arrival_sample']
        if len(str(tmp_p_t)) == 0:
            p_ar_sample = -1
        else:
            p_ar_sample = int(float(tmp_p_t))

        # arrival time of S wave
        tmp_s_t = self.meta.loc[self.meta.index[idx], 'trace_S_arrival_sample']
        if len(str(tmp_s_t)) == 0:
            s_ar_sample = -1
        else:
            s_ar_sample = int(float(tmp_s_t))
        
        labels = self.build_labels(waveforms, p_ar_sample, s_ar_sample)
        if self.is_train:
            return waveforms, labels
        else:
            ground_truth = np.array([p_ar_sample, s_ar_sample])
            if self.wave_length is not None:
                ground_truth[ground_truth >= self.wave_length] = -1
            return waveforms, labels, ground_truth

    def build_labels(self, waveforms, p_time, s_time):
        target = np.zeros(waveforms.shape[::-1])    # shaped wave_length x 3
        # target[:, 0] denotes to probability of noise, 1 of P wave and 2 of S wave
        if self.label_shape == 'gaussian':
            label_window = np.exp(
                -((np.arange(-self.label_width // 2, self.label_width // 2 + 1)) ** 2)
                / (2 * (self.label_width / 5) ** 2)
            )
        elif self.label_shape == 'triangle':
            label_window = 1 - np.abs(
                2 / self.label_width * (np.arange(-self.label_width // 2, self.label_width // 2 + 1))
            )
        elif self.label_shape == 'box':
            label_window = np.ones(self.label_width)
        else:
            print(f'Label shape "{self.label_shape}" should be gaussian, triangle or box.')
            raise

        for i, ar_t in enumerate([p_time, s_time]):
            if ar_t < 0:
                continue
            else:
                idx = ar_t
                if (idx - self.label_width // 2 >= 0) and (idx + self.label_width // 2 + 1 <= target.shape[0]):
                    target[idx - self.label_width // 2: idx + self.label_width // 2 + 1, i + 1] = label_window
        target[:, 0] = 1 - np.sum(target[:, 1:], axis=-1)
        return target.T     # shaped 3 x wave_length after transpose

