import tokenizers
import torch
from sequitur.models import LINEAR_AE
from sequitur import quick_train
from torch.utils.data import DataLoader

from dataset import LMDBDataset
import json


encoder = torch.load('./ckpts/encoder.pt')
print(encoder)
