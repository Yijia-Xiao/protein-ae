from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, embed_dim=128, hidden_size=[512, 512, 128]):
        super().__init__()
        self.hidden_list = [embed_dim] + hidden_size
        self.embedding = nn.Embedding(20, embedding_dim=embed_dim)
        self.layers = []
        for pre, lat in zip(self.hidden_list[:-1], self.hidden_list[1:]):
            self.layers.append(nn.Linear(pre, lat))
            # self.layers.append(nn.ReLU())
            self.layers.append(nn.SiLU())
        # del self.layers[-1]

        self.net = nn.Sequential(
            *self.layers
        )
        self.head = nn.Linear(self.hidden_list[-1], 1)

    def forward(self, x):
        # print(x)
        x = self.embedding(x)
        rep = self.net(x)
        # print(rep.shape)
        seq_rep = rep.mean(dim=-2)
        # print(seq_rep.shape)
        y = self.head(seq_rep)
        # print(y.shape)
        return y
