import argparse
import os

import time
from tqdm import tqdm

import torch
import numpy as np

from utils.data_utils import get_train_valid_loader
from utils.scheduler import WarmupCosineSchedule
from models.modeling import PhaseNet, EQTransformer
from models.criterion import evaluation, _loss
from models.config import PHASENET_CFG


def setup(args):
    if args.model == 'phasenet':
        model = PhaseNet(PHASENET_CFG)
    elif args.model == 'eqtransformer':
        model = EQTransformer()
    model = model.to(args.device)
    return args, model


def valid(args, model, valid_loader):
    print('\n')
    print('----------Validating----------')
    epoch_iterator = tqdm(valid_loader,
                          desc='Validating...',
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
            f'Validation {step}, loss = {loss}'
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

    print("Validation Results")
    print("P wave")
    print(f"mean: {mean_p}, var: {std_deviation_p}, Pr: {Pr_p}, Re: {Re_p}, F1: {F1_p}")
    print(f"Tp: {tot_Tp_p}, Fp: {tot_Fp_p}, Tn: {tot_Tn_p}, Fn: {tot_Fn_p}")
    print("S wave")
    print(f"mean: {mean_s}, var: {std_deviation_s}, Pr: {Pr_s}, Re: {Re_s}, F1: {F1_s}")
    print(f"Tp: {tot_Tp_s}, Fp: {tot_Fp_s}, Tn: {tot_Tn_s}, Fn: {tot_Fn_s}")
    
    return mean_p, mean_s


def train(args, model):
    train_loader, valid_loader = get_train_valid_loader(args)

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9)
    scheduler = WarmupCosineSchedule(optimizer=optimizer,
                                     warmup_steps=args.warmup_steps,
                                     t_total=args.num_steps)

    # Training!
    print('----------Training----------')
    print(f' Total training epochs: {args.num_steps}')
    print(f' Train batch size: {args.train_batch_size}')

    model.zero_grad()
    global_step = 0
    start_time = time.time()
    best_mean_p = np.inf
    best_mean_s = np.inf

    while True:
        model.train()

        epoch_iterator = tqdm(train_loader,
                              desc='Training (X/X steps) (loss = X.X)',
                              bar_format='{l_bar}{r_bar}',
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            # shaped batch_size x 3 (channels) x wave_length

            pred = model(x.float())
            loss = _loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            scheduler.step()
            optimizer.step()
            

            epoch_iterator.set_description(
                f'Training {global_step} / {args.num_steps}, loss = {loss}'
            )
            global_step += 1

            if global_step % args.valid_every == 0 or global_step == 1:
                with torch.no_grad():
                    model.eval()
                    mean_p, mean_s = valid(args, model, valid_loader)
                if not np.isnan(mean_p):
                    if abs(mean_p) < abs(best_mean_p):
                        best_mean_p = mean_p
                        save_model(args, model, 'best_mean_p')
                if not np.isnan(mean_s):
                    if abs(mean_s) < abs(best_mean_s):
                        best_mean_s = mean_s
                        save_model(args, model, 'best_mean_s')
                print(f'best mean so far: P-wave({best_mean_p}), S-wave({best_mean_s})')
                model.train()

            if global_step % args.num_steps == 0:
                break

        if global_step % args.num_steps == 0:
            break


def save_model(args, model, suffix=''):
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint_save_dir = os.path.join(args.result_save_root, f"{args.name}_{args.model}_checkpoint_{suffix}.bin")
    checkpoint = {'model': model_to_save.state_dict()}
    torch.save(checkpoint, checkpoint_save_dir)
    print('Saved model checkpoint to [DIR: %s]', args.result_save_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default_train')

    # args of file roots
    parser.add_argument('--dataset_root', type=str, default='../data')
    parser.add_argument('--result_save_root', type=str, default='result')

    # args of data
    parser.add_argument('--label_shape', type=str, default='gaussian')
    parser.add_argument('--wave_length', type=int, default=6001)
    parser.add_argument('--label_width', type=int, default=100)

    # args of training
    parser.add_argument('--model', type=str, choices=['phsenet', 'eqtransformer'], default='phasenet')
    parser.add_argument('--train_split', type=float, default=0.95)
    parser.add_argument('--valid_split', type=float, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--valid_every', type=int, default=1000)
    parser.add_argument('--valid_batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=3e-2)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--warmup_steps', type=int, default=500)

    args = parser.parse_args()

    # args of device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # args of files
    args.events_file = os.path.join(args.dataset_root, 'instance_events_counts.hdf5')
    args.events_info_file = os.path.join(args.dataset_root, 'metadata_instance_events_v2.csv')
    args.noise_file = os.path.join(args.dataset_root, 'instance_noise.hdf5')
    args.noise_info_file = os.path.join(args.dataset_root, 'metadata_instance_noise.csv')
    if not os.path.exists(args.result_save_root):
        os.mkdir(args.result_save_root)
    args.logger_file = os.path.join(args.result_save_root, 'logger.txt')

    # args of training
    args.test_split = 1 - args.train_split - args.valid_split
    assert 0 < args.test_split < 1
    assert 0 < args.train_split < 1
    assert 0 < args.valid_split < 1

    args, model = setup(args)

    train(args, model)


if __name__ == '__main__':
    main()
