from dataset import LMDBDataset
import json


data = []
def data_provider(split):
    dataset = LMDBDataset(f'data/fluorescence_{split}.lmdb')
    for item in dataset:
        data.append((item['primary'], item['log_fluorescence'].item()))

    # print(len(data))
    with open(f'./data/{split}.json', 'w') as f:
        json.dump(data, f)
    return data

train = data_provider('train')
valid = data_provider('valid')
test = data_provider('test')


# import lmdb
# env_db = lmdb.Environment('data/fluorescence_valid')
# txn = env_db.begin()
# print('hi')
# # The get function queries data by key value, and outputs None if the key value to be queried does not have corresponding data.
# for key, value in txn.cursor():
#     print(key, value)
# env_db.close()
