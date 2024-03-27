import os
import torch
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py

from utils.data_utils import get_test_loader
from models.modeling import ARPicker, PhaseNet, EQTransformer
from models.criterion import list_form_eval, _loss, evaluation


def setup(args):
    if args.model == 'ARPicker':
        model = ARPicker()
    elif args.model == 'PhaseNet':
        model = PhaseNet()
        state_dict = torch.load(
            os.path.join(args.result_save_root, args.checkpoint),
            map_location=args.device
        )['model']
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.eval()
    elif args.model == 'EQTransformer':
        model = EQTransformer(in_samples=args.wave_length)
        state_dict = torch.load(
            os.path.join(args.result_save_root, args.checkpoint),
            map_location=args.device
        )['model']
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.eval()
    return args, model


def test_traditional(args, model):
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

    all_pred_p = []
    all_pred_s = []
    all_truth_p = []
    all_truth_s = []

    print('----------Testing----------')
    for i in range(len(testing_meta)):
        if i % 100 == 0 or i == len(testing_meta) - 1:
            print('testing %.2f' % (i / len(testing_meta)) + '%')
        trace_name = testing_meta.loc[testing_meta.index[i], 'trace_name']
        waveforms = hdf_data['data'][trace_name]
        waveforms = np.array(waveforms)

        p_pick, s_pick = model(waveforms)
        all_pred_p.append(p_pick * 100)
        all_pred_s.append(s_pick * 100)

        tmp_p_t = testing_meta.loc[testing_meta.index[i], 'trace_P_arrival_sample']
        if len(str(tmp_p_t)) == 0:
            p_ar_sample = -1
        else:
            p_ar_sample = int(float(tmp_p_t))
        all_truth_p.append(p_ar_sample)

        # arrival time of S wave
        tmp_s_t = testing_meta.loc[testing_meta.index[i], 'trace_S_arrival_sample']
        if len(str(tmp_s_t)) == 0:
            s_ar_sample = -1
        else:
            s_ar_sample = int(float(tmp_s_t))
        all_truth_s.append(s_ar_sample)

    _mean_p, _std_deviation_p, _Pr_p = list_form_eval(all_pred_p, all_truth_p)
    _mean_s, _std_deviation_s, _Pr_s = list_form_eval(all_pred_s, all_truth_s)

    print("Testing Results")
    print(f"P wave: mean({_mean_p}), sigma({_std_deviation_p}), Pr({_Pr_p})")
    print(f"S wave: mean({_mean_s}), sigma({_std_deviation_s}), Pr({_Pr_s})")


def test(args, model):
    test_loader = get_test_loader(args)
    print('\n')
    print('----------Testing----------')
    epoch_iterator = tqdm(test_loader,
                          desc='Testing...',
                          bar_format='{l_bar}{r_bar}',
                          dynamic_ncols=True)
    tot_cnt_p = tot_cnt_s = 0
    tot_Tp_p = tot_Fp_p = tot_Tn_p = tot_Fn_p = 0
    tot_Tp_s = tot_Fp_s = tot_Tn_s = tot_Fn_s = 0
    tot_sum_p = tot_square_sum_p = 0.0
    tot_sum_s = tot_square_sum_s = 0.0

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y, ground_truth = batch
        with torch.no_grad():
            pred = model(x.float())
            loss = _loss(pred, y)
        
        epoch_iterator.set_description(
            f'Testing {step}, loss = {loss}'
        )

        cnt_p, sum_p, square_sum_p, Tp_p, Fp_p, Tn_p, Fn_p, cnt_s, sum_s, square_sum_s, Tp_s, Fp_s, Tn_s, Fn_s = evaluation(pred, ground_truth)
        tot_cnt_p += cnt_p
        tot_sum_p += sum_p
        tot_square_sum_p += square_sum_p
        tot_Tp_p += Tp_p
        tot_Fp_p += Fp_p
        tot_Tn_p += Tn_p
        tot_Fn_p += Fn_p
        tot_cnt_s += cnt_s
        tot_sum_s += sum_s
        tot_square_sum_s += square_sum_s
        tot_Tp_s += Tp_s
        tot_Fp_s += Fp_s
        tot_Tn_s += Tn_s
        tot_Fn_s += Fn_s
    
    mean_p = tot_sum_p / tot_cnt_p if tot_cnt_p > 0 else float('nan')
    std_deviation_p = (tot_square_sum_p / tot_cnt_p - mean_p * mean_p) ** 0.5 if tot_cnt_p > 0 else float('nan')
    Pr_p = tot_Tp_p / (tot_Tp_p + tot_Fp_p) if (tot_Tp_p + tot_Fp_p) > 0 else float('nan')
    Re_p = tot_Tp_p / (tot_Tp_p + tot_Fn_p) if (tot_Tp_p + tot_Fn_p) > 0 else float('nan')
    F1_p = 2 * Pr_p * Re_p / (Pr_p + Re_p) if (Pr_p + Re_p) > 0 else float('nan')
    
    mean_s = tot_sum_s / tot_cnt_s if tot_cnt_s > 0 else float('nan')
    std_deviation_s = (tot_square_sum_s / tot_cnt_s - mean_s * mean_s) ** 0.5 if tot_cnt_s > 0 else float('nan')
    Pr_s = tot_Tp_s / (tot_Tp_s + tot_Fp_s) if (tot_Tp_s + tot_Fp_s) > 0 else float('nan')
    Re_s = tot_Tp_s / (tot_Tp_s + tot_Fn_s) if (tot_Tp_s + tot_Fn_s) > 0 else float('nan')
    F1_s = 2 * Pr_s * Re_s / (Pr_s + Re_s) if (Pr_s + Re_s) > 0 else float('nan')

    print("Testing Results")
    print("P wave")
    print(f"mean: {mean_p}, var: {std_deviation_p}, Pr: {Pr_p}, Re: {Re_p}, F1: {F1_p}")
    print(f"Tp: {tot_Tp_p}, Fp: {tot_Fp_p}, Tn: {tot_Tn_p}, Fn: {tot_Fn_p}")
    print("S wave")
    print(f"mean: {mean_s}, var: {std_deviation_s}, Pr: {Pr_s}, Re: {Re_s}, F1: {F1_s}")
    print(f"Tp: {tot_Tp_s}, Fp: {tot_Fp_s}, Tn: {tot_Tn_s}, Fn: {tot_Fn_s}")
    
    return mean_p, mean_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default_test')
    
    # args of files
    parser.add_argument('--dataset_root', type=str, default='../data')
    parser.add_argument('--result_save_root', type=str, default='result')

    # args of testing
    parser.add_argument('--data_size', type=int, default=0)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--wave_length', type=int, default=12000)
    parser.add_argument('--model', type=str, choices=['ARPicker', 'PhaseNet', 'EQTransformer'], default='EQTransformer')
    parser.add_argument('--checkpoint', type=str, default='eqt_test_checkpoint.bin')

    args = parser.parse_args()

    # args of device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # args of files
    args.events_file = os.path.join(args.dataset_root, 'instance_events_counts.hdf5')
    args.events_info_file = os.path.join(args.dataset_root, 'metadata_instance_events_v2.csv')

    if args.data_size <= 0:
        args.data_size = None

    args, model = setup(args)
    
    if args.model == 'ARPicker':
        test_traditional(args, model)
    else:
        test(args, model)


if __name__ == '__main__':
    main()
