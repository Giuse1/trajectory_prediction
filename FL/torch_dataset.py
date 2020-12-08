import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
import numpy as np
import os
random.seed(0)


class UserDataset(Dataset):

    def __init__(self, idx, info, start_info):
        path = "data_ngsim_np/"
        self.trainset = np.load(f"{path}{idx}_r50.npy")

        self.start_arr = start_info[str(idx)]
        self.info = info
        self.window = 100

    def __len__(self):
        return len(self.start_arr)

    def __getitem__(self, idx_sample):
        s = self.trainset[self.start_arr[idx_sample]:self.start_arr[idx_sample] + 100, :]
        seq = torch.from_numpy(s[:int(0.9 * self.window), 2:])
        target = torch.from_numpy(s[int(0.9 * self.window):, 2:4])
        fixed = self.info[self.info['index'] == s[0, 1]].values[0, 2:5].astype(float)
        fixed = torch.from_numpy(fixed)
        sample = {'seq': seq, 'target': target, 'fixed': fixed}

        return sample


def get_loaders(info_dataset, batch_size=8, shuffle=True):

    p = '/content/drive/MyDrive/general_data/'
    with open(p + 'start_arr.json', 'r') as fp:
        start_arr = json.load(fp)

    training_list = []
    path = "/content/drive/MyDrive/data_ngsim_np/"

    list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    total_num_users = len(list_files)
    training_list = {}
    for i in range(total_num_users):
        data = UserDataset(idx=i, info=info_dataset, start_info=start_arr)
        try:
            loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
            training_list[i] = loader
        except:
            pass

    #num_training = int(0.9 * len(training_list))
    #test_list = training_list[num_training:]
    #print("test set lenght:" + str(len(test_list)))

    #training_list = training_list[:num_training]
    #print("training set lenght:" + str(len(training_list)))

    return training_list


def get_correct_ids():
    p = '/content/drive/MyDrive/general_data/'
    with open(p + 'start_arr.json', 'r') as fp:
        start_arr = json.load(fp)

    for k, v in list(start_arr.items()):
        if start_arr[k] == []:
            del start_arr[k]

    correct_vehicles_ids = list(start_arr.keys())
    correct_vehicles_ids = [float(x) for x in correct_vehicles_ids]

    return correct_vehicles_ids
#funzione a parte per correct ids
#fare tutti i dataloaders
#random sampling da correct ids
#per gruppi non random -> togliere dal dizionario elementi non apprtenenti a correct ids