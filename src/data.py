import os
import json
from numpy import dtype
import torch
from torch.utils.data import Dataset


class Tokenizer(object):
    def __init__(self, vocab='ARNDCQEGHILKMFPSTWYV'):
        self.vocab = vocab
        self.id_char = dict()
        self.char_id = dict()
        for i in range(len(self.vocab)):
            self.id_char[i] = self.vocab[i]
        for k, v in self.id_char.items():
            self.char_id[v] = k

    def __call__(self, x):
        return list(map(lambda c: self.char_id[c], x))


class FDataset(Dataset):
    def __init__(self, split, root='./data') -> None:
        super().__init__()
        assert split in ['train', 'valid', 'test']
        raw_data = json.load(open(os.path.join(root, f'{split}.json'), 'r'))
        self.STD_LEN = 237
        self.data = [item for item in raw_data if len(item[0]) == self.STD_LEN]
        # print(self.data[0])
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.IntTensor(self.tokenizer(self.data[index][0])), torch.FloatTensor([self.data[index][1]])
