import torch
import numpy as np


def _loss(y, labels):
    # y (predicted) and labels (ground truth) are shaped batch_size x 3 x wave_length
    # note that y should be a softmax result
    h = labels * torch.log(y+1e-5)
    h = h.mean(-1).sum(-1)
    h = h.mean()
    return -h


def evaluation(y, ground_truth):
    # y (predicted) are shaped batch_size x 3 x wave_length
    # note that y should be a softmax result
    ground_truth = ground_truth.cpu().numpy()
    y = y.cpu().numpy()
    batch_size, _, _ = y.shape

    max_val = y.max(axis=2)[:, 1:]
    t_pred = y.argmax(axis=2)[:, 1:]  # ignore noise and find peak
    # max_val and t_pred are shaped batch_size x 2
    t_pred[max_val < 0.5] = -1
    # peak probabilities above 0.5 are counted as positive picks

    # ground_truth are shaped batch_size x 2
    # GT[:, 0] represents the time of P wave and GT[:, 1] the S wave

    assert ground_truth.shape == t_pred.shape

    truth_p = ground_truth[:, 0]
    truth_s = ground_truth[:, 1]
    pred_p = t_pred[:, 0]
    pred_s = t_pred[:, 1]
    delta_p = truth_p - pred_p
    delta_s = truth_s - pred_s

    DELTA_T = 0.5 * 100   # 100Hz sampling rate
    # Arrival-time residuals that are less than 0.1s are counted as true positives

    Tp_p = Fp_p = Tn_p = Fn_p = 0
    TPP = []
    
    Tp_s = Fp_s = Tn_s = Fn_s = 0
    TPS = []
    
    for i in range(delta_p.shape[0]):
        if truth_p[i] >= 0:
            if pred_p[i] >= 0:
                TPP.append(i)
            if pred_p[i] >= 0 and abs(delta_p[i]) < DELTA_T:
                Tp_p += 1
            else:
                Fp_p += 1
        else:
            if pred_p[i] < 0:
                Tn_p += 1
            else:
                Fn_p += 1

    for i in range(delta_s.shape[0]):
        if truth_s[i] >= 0:
            if pred_s[i] >= 0:
                TPS.append(i)
            if pred_s[i] >= 0 and abs(delta_s[i]) < DELTA_T:
                Tp_s += 1
            else:
                Fp_s += 1
        else:
            if pred_s[i] < 0:
                Tn_s += 1
            else:
                Fn_s += 1

    delta_p_p = delta_p[TPP]
    delta_p_s = delta_s[TPS]
    cnt_p = len(delta_p_p)
    cnt_s = len(delta_p_s)
    sum_p = delta_p_p.sum() / 100
    sum_s = delta_p_s.sum() / 100
    square_sum_p = (delta_p_p * delta_p_p).sum() / (100 * 100)
    square_sum_s = (delta_p_s * delta_p_s).sum() / (100 * 100)

    # _Pr = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    # _Re = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    # _F1 = 2 * _Pr * _Re / (_Pr + _Re) if _Pr + _Re > 0 else 0.0

    return cnt_p, sum_p, square_sum_p, Tp_p, Fp_p, Tn_p, Fn_p, cnt_s, sum_s, square_sum_s, Tp_s, Fp_s, Tn_s, Fn_s


def list_form_eval(pred, truth):
    # used for evaluatin of traditional method
    assert len(pred) == len(truth)
    pred = np.array(pred, dtype=float)
    truth = np.array(truth, dtype=float)
    _pred = pred[truth >= 0]
    _truth = truth[truth >= 0]
    _delta = _truth - _pred
    
    DELTA_T = [0.1 * 100, 0.3 * 100, 0.5 * 100, 1.0 * 100, 2.0 * 100]   # 100Hz sampling rate
    # Arrival-time residuals that are less than 0.1s are counted as true positives
    # But for traditional methos, 0.1s might be too less and the Pr value can be below 0.5!
    
    Pr_lst = []
    for dt in DELTA_T:
        true_positive = len(_delta[abs(_delta) <= dt])
        _mean = _delta.mean() / 100
        _std_deviation = np.sqrt(_delta.var()) / 100
        _Pr = true_positive / _delta.shape[0] if _delta.shape[0] > 0 else np.nan
        Pr_lst.append((f'threshold = {dt}s', _Pr))
    return _mean, _std_deviation, Pr_lst

