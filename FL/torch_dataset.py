import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import random
import json
import os

random.seed(0)


class UserDataset(Dataset):

    def __init__(self, idx, info, start_info):
        path = "/content/drive/MyDrive/data_ngsim/"
        self.trainset = pd.read_csv(f"{path}{idx}_r50.csv")
        scaler = MinMaxScaler(feature_range=(-5, 5))

        self.trainset[self.trainset.columns[2:]] = scaler.fit_transform(self.trainset[self.trainset.columns[2:]])

        self.start_arr = start_info[str(idx)]
        self.info = info
        self.window = 100


    def __len__(self):

        return len(self.start_arr)

    def __getitem__(self, idx):

        s = self.trainset.iloc[self.start_arr[idx]:self.start_arr[idx]+100]
        seq = torch.from_numpy(s.iloc[:int(0.9 * self.window)].values)
        target = torch.from_numpy(s.iloc[int(0.9 * self.window):][["Local_X", "Local_Y"]].values)
        fixed = self.info[self.info['index']==s['info'].unique()[0]].values[0][2:5].astype(float)
        fixed = torch.from_numpy(fixed)
        sample = {'seq': seq, 'target': target, 'fixed': fixed}

        return sample


def get_loaders(info_dataset, batch_size=8, shuffle=True):
    training_list = []
    import os.path
    p = (os.path.dirname(__file__))
    with open(p+'/../dict.json', 'r') as fp:
        start_arr = json.load(fp)

    for k, v in list(start_arr.items()):
        if start_arr[k] == []:
            del start_arr[k]

    import time
    for i in start_arr.keys():
        t = time.time()
        data = UserDataset(idx=i, info=info_dataset, start_info=start_arr)
        print(time.time()-t)
        t = time.time()

        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        print(time.time()-t)
        t = time.time()

        training_list.append(loader)
        print(time.time()-t)


        
        #print(i)
        
    num_training = int(0.9 * len(training_list))
    test_list = training_list[num_training:]
    print(len(test_list))
    print(len(training_list))
    training_list = training_list[:num_training]
    print(len(training_list))

    return training_list, test_list
