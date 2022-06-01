from fast_soft_sort.pytorch_ops import soft_rank
import torch
import pandas as pd
import numpy as np

def corrcoef(target, pred):
    # np.corrcoef in torch from @mdo
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
    regularization_strength=1.0,
):
    # fast_soft_sort uses 1-based indexing, divide by len to compute percentage of rank
    pred = soft_rank(
        pred,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    return corrcoef(target, pred / pred.shape[-1])


def numerai_spearman(target, pred):
    # spearman used for numerai CORR
    return np.corrcoef(target, pred.rank(pct=True, method="first"))[0, 1]

# my spearman requires having batch dimension as first.
pred = torch.rand(1, 10, requires_grad=True)
target = torch.rand(1, 10)

print(pred.shape, target.shape)

print("Numerai CORR", numerai_spearman(
    pd.Series(target[0].detach().numpy()),
    pd.Series(pred[0].detach().numpy()),
))

s = spearman(target, pred, regularization_strength=1e-3)
gradient = torch.autograd.grad(s, pred)[0]
print("Differentiable CORR", s.item())


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


print(spearman_correlation(target[0], pred[0]))


from scipy.stats import stats
r, p = stats.spearmanr(target[0].detach().numpy(), pred[0].detach().numpy())
print(r)