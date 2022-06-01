from fast_soft_sort.pytorch_ops import soft_rank
import torch
import pandas as pd
import numpy as np


def corrcoef(target, pred):
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()


def spearman(
        target,
        pred,
        regularization="l2",
        regularization_strength=1.0):
    # fast_soft_sort uses 1-based indexing, divide by len to compute percentage of rank
    pred = soft_rank(
        pred,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    return corrcoef(target, pred / pred.shape[-1])


def np_spearman(target, pred):
    # spearman used for numerai CORR
    return np.corrcoef(target, pred.rank(pct=True, method="first"))[0, 1]
