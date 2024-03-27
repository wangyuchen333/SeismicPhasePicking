import os

import h5py
import numpy as np
import pandas as pd

from .dataset import SeismicDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms


def split_train_validation(args, dataset_length):
    idx_lst = np.array(np.arange(dataset_length), dtype=int)
    np.random.shuffle(idx_lst)
    training_idx = idx_lst[:int(args.train_split * dataset_length)]
    validation_idx = idx_lst[int(args.train_split * dataset_length):
                             int((args.valid_split + args.train_split) * dataset_length)]
    testing_idx = idx_lst[int((args.valid_split + args.train_split) * dataset_length):]
    np.save(os.path.join(args.result_save_root, 'testing_meta.npy'), testing_idx)
    return training_idx, validation_idx


def get_train_valid_loader(args):
    meta_df = pd.read_csv(args.events_info_file,
                          keep_default_na=False,
                          dtype={'station_location_code': object,
                                 'source_mt_eval_mode': object,
                                 'source_mt_status': object,
                                 'source_mechanism_strike_dip_rake': object,
                                 'source_mechanism_moment_tensor': object,
                                 'trace_P_arrival_time': object,
                                 'trace_S_arrival_time': object
                                 }, low_memory=False)
    training_idx, validation_idx = split_train_validation(args, len(meta_df))

    training_meta = meta_df.iloc[training_idx, :].reset_index(drop=False)
    validation_meta = meta_df.iloc[validation_idx, :].reset_index(drop=False)

    hdf_data = h5py.File(args.events_file, 'r')
    training_data = SeismicDataset(
        meta=training_meta,
        hdf_data=hdf_data,
        label_shape='gaussian',
        label_width=args.label_width,
        wave_length=args.wave_length
    )
    validation_data = SeismicDataset(
        meta=validation_meta,
        hdf_data=hdf_data,
        is_train=False,
        label_width=args.label_width,
        wave_length=args.wave_length
    )

    training_sampler = RandomSampler(training_data)
    validation_sampler = SequentialSampler(validation_data)

    training_loader = DataLoader(
        training_data,
        sampler=training_sampler,
        batch_size=args.train_batch_size,
        drop_last=True,
        pin_memory=True)
    validation_loader = DataLoader(
        validation_data,
        sampler=validation_sampler,
        batch_size=args.valid_batch_size,
        pin_memory=True)
    return training_loader, validation_loader


def get_test_loader(args):
    meta_df = pd.read_csv(args.events_info_file,
                          keep_default_na=False,
                          dtype={'station_location_code': object,
                                 'source_mt_eval_mode': object,
                                 'source_mt_status': object,
                                 'source_mechanism_strike_dip_rake': object,
                                 'source_mechanism_moment_tensor': object,
                                 'trace_P_arrival_time': object,
                                 'trace_S_arrival_time': object
                                 }, low_memory=False)
    NPY_FILE_DIR = os.path.join(args.result_save_root, 'testing_meta.npy')
    testing_idx = np.load(NPY_FILE_DIR)[:args.data_size]
    testing_meta = meta_df.iloc[testing_idx, :].reset_index(drop=False)
    hdf_data = h5py.File(args.events_file, 'r')
    testing_data = SeismicDataset(
        meta=testing_meta,
        hdf_data=hdf_data,
        is_train=False,
        wave_length=args.wave_length
    )

    testing_sampler = SequentialSampler(testing_data)
    testing_loader = DataLoader(
        testing_data,
        sampler=testing_sampler,
        batch_size=args.test_batch_size,
        pin_memory=True)
    return testing_loader
