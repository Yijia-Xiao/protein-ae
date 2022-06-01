import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.data import FDataset
from src.model import Regressor
from src.utils import spearman
from src.utils import np_spearman
from scipy.stats import stats
from torch.utils.tensorboard import SummaryWriter


import argparse

parser = argparse.ArgumentParser(description='Regressor')

parser.add_argument('--loss', choices=['mse', 'spear'], help='appoint the type of loss')
parser.add_argument('--optim', choices=['sgd', 'adamw'], help='appoint the type of optim')
args = parser.parse_args()
# print(args.loss)

writer = SummaryWriter(
    log_dir=f'./logs/{args.loss}-mean-{args.optim}')

device = torch.device('cpu')

train_data = FDataset('train')
train_loder = DataLoader(train_data, batch_size=len(train_data), shuffle=True)

valid_data = FDataset('valid')
valid_loder = DataLoader(valid_data, batch_size=len(valid_data), shuffle=False)

test_data = FDataset('test')
test_loder = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

model = Regressor().to(device)

if args.optim == 'sgd':
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
elif args.optim == 'adamw':
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

loss_mse = nn.MSELoss()

NUM_EPOCH = 500

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


import tqdm
for epoch in tqdm.tqdm(range(NUM_EPOCH)):
    model.train()
    running_loss = 0.0
    for X, Y in train_loder:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        # print(pred, Y)
        # print(spearman(pred.view(1, -1), Y.view(1, -1)))
        if args.loss == 'spear':
            loss = -spearman(pred.view(1, -1), Y.view(1, -1))
        elif args.loss == 'mse':
            loss = loss_mse(pred.view(1, -1), Y.view(1, -1))
        running_loss += loss.detach().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        spr = stats.spearmanr(pred.detach().numpy(), Y.detach().numpy())
    writer.add_scalar('train-loss', running_loss / len(train_data), epoch)
    writer.add_scalar('train-spear', spr[0], epoch)
    print('train spearmanr =', spr[0])

    for X, Y in valid_loder:
        model.eval()
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        spr = stats.spearmanr(pred.detach().numpy(), Y.detach().numpy())
        print('valid spearmanr =', spr[0])

        writer.add_scalar('valid-spear', spr[0], epoch)
        # print('train spearmanr =', spr[0])

    for X, Y in test_loder:
        model.eval()
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        spr = stats.spearmanr(pred.detach().numpy(), Y.detach().numpy())
        print('test spearmanr =', spr[0])

        writer.add_scalar('test-spear', spr[0], epoch)
        # print('train spearmanr =', spr[0])
