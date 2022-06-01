import torch
# import spacy
from torch.utils.data import DataLoader

from dataset import LMDBDataset
import json


data = []
STD_LEN = 237

def data_provider(split):
    num_discard = 0
    dataset = LMDBDataset(f'data/fluorescence_{split}.lmdb')
    for item in dataset:
        if len(item['primary']) == STD_LEN:
            data.append((item['primary'], item['log_fluorescence'].item()))
        else:
            num_discard += 1

    # print(len(data))
    with open(f'./data/{split}.json', 'w') as f:
        json.dump(data, f)
    print(f'{split}, num_discard = {num_discard}')

    return data


class Tokenizer(object):
    def __init__(self, vocab='ARNDCQEGHILKMFPSTWYV'):
        self.vocab = vocab
        self.id_char = dict()
        self.char_id = dict()
        for i in range(len(self.vocab)):
            self.id_char[i] = self.vocab[i]
        for k, v in self.id_char.items():
            self.char_id[v] = k
        # print(self.char_id, self.id_char)

    def __call__(self, x):
        # print(list(map(lambda c: self.char_id[c], x)))
        # print(x[0])
        return list(map(lambda c: self.char_id[c], x[0]))

# spacy.tokenizer.Tokenizer()
tokenizer = Tokenizer()
# train_seqs = [torch.randn(4) for _ in range(100)]

# print('train', tokenizer('TRAIN'))
train = data_provider('train')
valid = data_provider('valid')
test = data_provider('test')

train_seqs = []
valid_seqs = []
test_seqs = []

for sample in train:
    train_seqs.append(torch.Tensor(tokenizer(sample)))

for sample in valid:
    valid_seqs.append(torch.Tensor(tokenizer(sample)))

for sample in test:
    test_seqs.append(torch.Tensor(tokenizer(sample)))

# train_seqs = DataLoader(train_seqs, batch_size=16)

def train():
    from sequitur.models import LINEAR_AE
    from sequitur import quick_train
    # encoder, decoder, _, _ = quick_train(LINEAR_AE, train_seqs, encoding_dim=128, denoise=True, lr=1e-4, epochs=20)
    encoder, decoder, _, _ = quick_train(LINEAR_AE, train_seqs, encoding_dim=128, denoise=True, lr=1e-4, epochs=100)

    torch.save(encoder, './ckpts/encoder.pt')
    torch.save(decoder, './ckpts/decoder.pt')
    torch.save(_, './ckpts/_.pt')

    # print(encoder(torch.randn(237)))

# train()


def prov_data():
    encoder = torch.load('./ckpts/encoder.pt')
    encoder.eval()
    # enc = encoder(test_seqs[0])

    # print(enc)
    # train_samples = train_seqs + valid_seqs
    # print(encoder(test_seqs[i]))

    samples = json.load(open('./data/test.json', 'r'))

    embeds = [encoder(x) for x in test_seqs]
    labels = [i[1] for i in samples]
    return embeds, labels

# print(len(embeds), len(labels))


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
class FDataset(Dataset):
    def __init__(self):
        embeds, labels = prov_data()
        self.embeds = embeds
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        # return torch.FloatTensor(self.embeds[index]), self.labels[index]
        # print(self.embeds[index], self.labels[index])
        return self.embeds[index].detach(), self.labels[index]
import argparse
parser = argparse.ArgumentParser(description='Regressor')

parser.add_argument('--loss', choices=['mse', 'spear'], help='appoint the type of loss')

parser.add_argument('--optim', choices=['sgd', 'adamw'], help='appoint the type of optim')
parser.add_argument('--act', choices=['relu', 'silu'], help='appoint the type of act')

args = parser.parse_args()

class Regressor(nn.Module):
    def __init__(self, input_dim=128, hidden_size=[512, 512, 128]):
        super().__init__()
        self.hidden_list = [input_dim] + hidden_size
        # self.embedding = nn.Embedding(20, embedding_dim=embed_dim)
        self.layers = []
        for pre, lat in zip(self.hidden_list[:-1], self.hidden_list[1:]):
            self.layers.append(nn.Linear(pre, lat))
            # self.layers.append(nn.ReLU())
            self.layers.append(nn.SiLU() if args.act == 'silu' else nn.ReLU())
        # del self.layers[-1]

        self.net = nn.Sequential(
            *self.layers
        )
        self.head = nn.Linear(self.hidden_list[-1], 1)

    def forward(self, x):
        # print(x)
        # x = self.embedding(x)
        rep = self.net(x)
        # print(rep.shape)
        # seq_rep = rep.mean(dim=-2)
        # print(seq_rep.shape)
        y = self.head(rep)
        # print(y.shape)
        return y


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from scipy.stats import stats

# device = torch.device('cuda:0')
device = torch.device('cpu')
train_dataset = FDataset()
train_loder = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

model = Regressor().to(device)

loss_mse = nn.MSELoss()

from src.utils import spearman


from torch.utils.tensorboard import SummaryWriter




writer = SummaryWriter(
    log_dir=f'./logs/ae/{args.loss}-{args.optim}-{args.act}')
if args.optim == 'sgd':
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
elif args.optim == 'adamw':
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


NUM_EPOCH = 6000
import tqdm
model.train()
for epoch in tqdm.tqdm(range(NUM_EPOCH)):
    running_loss = 0.0
    for X, Y in train_loder:
        X, Y = X.to(device).float(), Y.to(device).float()
        pred = model(X)

        # optimizer.zero_grad()
        # loss = loss_mse(pred.view(1, -1), Y.view(1, -1))
        # # print(loss)
        # running_loss += loss.item()
        # loss.backward()
        # optimizer.step()

        # spr = stats.spearmanr(pred.cpu().detach().numpy(), Y.cpu().detach().numpy())
        # print(spr)

        if args.loss == 'spear':
            loss = -spearman(pred.view(1, -1), Y.view(1, -1))
        elif args.loss == 'mse':
            loss = loss_mse(pred.view(1, -1), Y.view(1, -1))
        running_loss += loss.detach().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        spr = stats.spearmanr(pred.detach().numpy(), Y.detach().numpy())

    writer.add_scalar('train-loss', running_loss / len(train_dataset), epoch)
    writer.add_scalar('train-spear', spr[0], epoch)
    print('train spearmanr =', spr[0])

    # writer.add_scalar('train-loss', running_loss / len(train_data), epoch)
    # writer.add_scalar('train-spear', spr[0], epoch)
    # print('train spearmanr =', spr[0])

    # for X, Y in valid_loder:
    #     model.eval()
    #     X, Y = X.to(device), Y.to(device)
    #     pred = model(X)
    #     spr = stats.spearmanr(pred.detach().numpy(), Y.detach().numpy())
    #     print('valid spearmanr =', spr[0])

    #     writer.add_scalar('valid-spear', spr[0], epoch)
    #     # print('train spearmanr =', spr[0])

    # for X, Y in test_loder:
    #     model.eval()
    #     X, Y = X.to(device), Y.to(device)
    #     pred = model(X)
    #     spr = stats.spearmanr(pred.detach().numpy(), Y.detach().numpy())
    #     print('test spearmanr =', spr[0])

    #     writer.add_scalar('test-spear', spr[0], epoch)
    #     # print('train spearmanr =', spr[0])
