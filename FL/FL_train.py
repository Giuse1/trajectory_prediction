from FL.FL_user import User
import copy
import torch
import torch.nn as nn
from FL.torch_dataset import get_loaders, get_correct_ids
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import random
random.seed(0)


def train_model(global_model, criterion, num_rounds, local_epochs, num_users, batch_size, learning_rate):
    train_loss = []
    val_loss = []
    info = pd.read_csv("/content/drive/MyDrive/general_data/correct_info.csv").drop(["Unnamed: 0"], axis=1)

    info["new"] = info["index"].astype(str) + '_' + info["v_length"].astype(str) + '_' + info["v_Width"].astype(
        str) + '_' + info["v_Class"].astype(str)

    users_ids = get_correct_ids()
    all_list = get_loaders(batch_size=batch_size, shuffle=True, info_dataset=info)
    total_num_users = len(users_ids)
    training_ids = random.sample(users_ids, int(0.9*total_num_users))
    test_ids = list(set(users_ids) - set(training_ids))
    print("total users: " +str(total_num_users))
    print("training set lenght:" + str(len(training_ids)))
    print("test set lenght:" + str(len(test_ids)))

    scaler_list = []
    path = "/content/drive/MyDrive/data_ngsim/"
    for i in test_ids[:50]:
        tmp = pd.read_csv(path+str(int(i))+"_r50.csv")[["diff_Local_X", "diff_Local_Y"]]
        scaler = MinMaxScaler(feature_range=(-5, 5))
        scaler.fit(tmp)
        scaler_list.append(scaler)

    for round in range(num_rounds):
        print('-' * 10)
        print('Epoch {}/{}'.format(round, num_rounds - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                local_weights = []
                samples_per_client = []
                total_data = 0
                total_loss = 0
                random_list = random.sample(training_ids, num_users)

                for idx in random_list:
                    #print(idx)
                    local_model = User(dataloader=all_list[idx], id=idx, criterion=criterion,
                                              local_epochs=local_epochs, learning_rate=learning_rate)
                    w, local_loss, total_local_data, local_total = local_model.update_weights(
                        model=copy.deepcopy(global_model).float(), epoch=round)
                    total_data += total_local_data
                    total_loss += local_loss
                    local_weights.append(copy.deepcopy(w))
                    samples_per_client.append(local_total)
                print('{} Loss: {:.4f}'.format(phase, total_loss/total_data))

                global_weights = average_weights(local_weights, samples_per_client)
                global_model.load_state_dict(global_weights)

            else:
                val_loss_r = model_evaluation(model=global_model.float(), dataloader_list=all_list, indeces=test_ids, scaler_list=scaler_list)


                val_loss.append(val_loss_r)
                print('{} Loss: {:.4f}'.format(phase, val_loss_r))

    return train_loss, val_loss,


def train_model_aggregated(global_model, criterion, num_rounds, local_epochs, num_users, users_per_group, batch_size,
                           learning_rate, mode):
    train_loss = []
    val_loss = []
    info = pd.read_csv("/content/drive/MyDrive/general_data/correct_info.csv").drop(["Unnamed: 0"], axis=1)

    info["new"] = info["index"].astype(str) + '_' + info["v_length"].astype(str) + '_' + info["v_Width"].astype(
        str) + '_' + info["v_Class"].astype(str)

    users_ids = get_correct_ids()
    all_list = get_loaders(batch_size=batch_size, shuffle=True, info_dataset=info)
    total_num_users = len(users_ids)
    training_ids = random.sample(users_ids, int(0.9 * total_num_users))
    test_ids = list(set(users_ids) - set(training_ids))
    print("total users: " + str(total_num_users))
    print("training set lenght:" + str(len(training_ids)))
    print("test set lenght:" + str(len(test_ids)))

    with open('/content/drive/MyDrive/general_data/distances.json', 'r') as fp:
        distances_dict = json.load(fp)

    scaler_list = []
    path = "/content/drive/MyDrive/data_ngsim/"
    for i in test_ids[:50]:
        tmp = pd.read_csv(path+str(int(i))+"_r50.csv")[["diff_Local_X", "diff_Local_Y"]]
        scaler = MinMaxScaler(feature_range=(-5,5))
        scaler.fit(tmp)
        scaler_list.append(scaler)

    num_groups = int(num_users / users_per_group)
    for round in range(num_rounds):
        print('-' * 10)
        print('Epoch {}/{}'.format(round, num_rounds - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                local_weights = []
                samples_per_client = []
                total_data = 0
                total_loss = 0

                if mode =="hybrid_random":
                    random_list = random.sample(training_ids, num_users)
                elif mode =="hybrid_non_random":
                    random_list = get_nonrandom_ids(distances_dict, training_ids, num_groups, users_per_group)

                for i in range(int(num_groups)):
                    for j in range(users_per_group):
                        idx = random_list[j + i * users_per_group]
                        local_model = User(dataloader=all_list[idx], id=idx, criterion=criterion,
                                                  local_epochs=local_epochs, learning_rate=learning_rate)

                        if j == 0:
                            w, local_loss, total_local_data, local_total = local_model.update_weights(
                                model=copy.deepcopy(global_model).float(), epoch=round)
                            samples_per_client.append(local_total)
                        else:
                            model_tmp = copy.deepcopy(global_model)
                            model_tmp.load_state_dict(w)
                            w, local_loss, total_local_data, local_total = local_model.update_weights(
                                model=model_tmp.float(), epoch=round)
                            samples_per_client[i] += local_total
                    total_data += total_local_data
                    total_loss += local_loss
                    local_weights.append(copy.deepcopy(w))

                print('{} Loss: {:.4f}'.format(phase, total_loss/total_data))
                global_weights = average_weights(local_weights, samples_per_client)
                global_model.load_state_dict(global_weights)

            else:
                x_loss, y_loss = model_evaluation(model=global_model.float(), dataloader_list=all_list, indeces=test_ids, scaler_list=scaler_list)

                #val_loss.append(val_loss_r)
                print('{} x_loss: {:.4f} y_loss: {:.4f}'.format(phase, x_loss, y_loss))

    return train_loss, val_loss


def train_model_aggregated_small_groups(global_model, criterion, num_rounds, local_epochs, num_users, users_per_group, batch_size,
                           learning_rate):
    train_loss = []
    val_loss = []
    info = pd.read_csv("/content/drive/MyDrive/general_data/correct_info.csv").drop(["Unnamed: 0"], axis=1)

    info["new"] = info["index"].astype(str) + '_' + info["v_length"].astype(str) + '_' + info["v_Width"].astype(
        str) + '_' + info["v_Class"].astype(str)

    users_ids = get_correct_ids()
    all_list = get_loaders(batch_size=batch_size, shuffle=True, info_dataset=info)
    total_num_users = len(users_ids)
    training_ids = random.sample(users_ids, int(0.9 * total_num_users))
    test_ids = list(set(users_ids) - set(training_ids))
    print("total users: " + str(total_num_users))
    print("training set lenght:" + str(len(training_ids)))
    print("test set lenght:" + str(len(test_ids)))

    with open('/content/drive/MyDrive/general_data/distances.json', 'r') as fp:
        distances_dict = json.load(fp)

    scaler_list = []
    path = "/content/drive/MyDrive/data_ngsim/"
    for i in test_ids[:50]:
        tmp = pd.read_csv(path + str(int(i)) + "_r50.csv")[["diff_Local_X", "diff_Local_Y"]]
        scaler = MinMaxScaler(feature_range=(-5, 5))
        scaler.fit(tmp)
        scaler_list.append(scaler)


    num_groups = int(num_users / users_per_group)
    for round in range(num_rounds):
        print('-' * 10)
        print('Epoch {}/{}'.format(round, num_rounds - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                local_weights = []
                samples_per_client = []
                total_data = 0
                total_loss = 0

                list_of_list = get_nonrandom_ids_small_groups(distances_dict, training_ids, num_groups, users_per_group)

                for i, l in enumerate(list_of_list):
                    for j, idx in enumerate(l):

                        local_model = User(dataloader=all_list[idx], id=idx, criterion=criterion,
                                                  local_epochs=local_epochs, learning_rate=learning_rate)

                        if j == 0:
                            w, local_loss, total_local_data, local_total = local_model.update_weights(
                                model=copy.deepcopy(global_model).float(), epoch=round)
                            samples_per_client.append(local_total)
                        else:
                            model_tmp = copy.deepcopy(global_model)
                            model_tmp.load_state_dict(w)
                            w, local_loss, total_local_data, local_total = local_model.update_weights(
                                model=model_tmp.float(), epoch=round)
                            samples_per_client[i] += local_total
                    total_data += total_local_data
                    total_loss += local_loss
                    local_weights.append(copy.deepcopy(w))

                print('{} Loss: {:.4f}'.format(phase, total_loss/total_data))
                global_weights = average_weights(local_weights, samples_per_client)
                global_model.load_state_dict(global_weights)
                # if round%10 == 0 and round!=0:
                #     torch.save(global_model.state_dict(), '/content/drive/MyDrive/lstm.pth')


            else:
                x_loss, y_loss = model_evaluation(model=global_model.float(), dataloader_list=all_list, indeces=test_ids, scaler_list=scaler_list)

                #val_loss.append(val_loss_r)
                print('{} x_loss: {:.4f} y_loss: {:.4f}'.format(phase, x_loss, y_loss))

    return train_loss, val_loss


def model_evaluation(model, dataloader_list, indeces, scaler_list):
    with torch.no_grad():
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_loss = 0
        local_total = 0
        y_total_error = 0
        x_total_error = 0
        for j, ind in enumerate(indeces[:50]):
            #print(ind)
            scaler = scaler_list[j]
            dataloader = dataloader_list[ind]
            for sample in dataloader:
                seq = sample["seq"].float().to(device)
                target = sample["target"].float().to(device)
                fixed = sample["fixed"].float().to(device)

                target_pred = model(seq, fixed)

                #print(target)
                scaled_target = [scaler.inverse_transform(x.detach().cpu().numpy()) for x in target]
                #print(scaled_target)
                #print(c)

                scaled_pred = [scaler.inverse_transform(x.detach().cpu().numpy()) for x in target_pred]
                scaled_target = np.dstack(scaled_target)
                scaled_pred = np.dstack(scaled_pred)
                #print(scaled_pred.shape)
                x_error = np.abs(scaled_target[:, 0, :] - scaled_pred[:, 0])
                y_error = np.abs(scaled_target[:, 1, :] - scaled_pred[:, 1])

                x_total_error += np.sum(x_error)
                y_total_error += np.sum(y_error)
                local_total += x_error.size
                #print(x_error.shape)
                #print(y_error.shape)
                #print(x_total_error)
                #print(y_total_error)
                #print(x_error.size)
                #print(c)


        return x_total_error/local_total, y_total_error/local_total


def average_weights(w, samples_per_client):

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[key] = torch.true_divide(w[i][key], 1 / samples_per_client[i])
            else:
                w_avg[key] += torch.true_divide(w[i][key], 1 / samples_per_client[i])
        w_avg[key] = torch.true_divide(w_avg[key], sum(samples_per_client))
    return w_avg

def get_nonrandom_ids(d, correct_vehicles_ids, num_groups, users_per_group):

    l = []
    vehicles_ids = [int(x) for x in correct_vehicles_ids]
    vehicles_ids.remove(2595)
    for g in range(num_groups):
        to_append = random.sample(vehicles_ids, 1)[0]
        l.append(int(to_append))
        vehicles_ids.remove(to_append)

        for i in range(users_per_group - 1):

            j = 0
            tmp_dict = d[str(int(to_append))]
            tmp = tmp_dict[j]
            while tmp in l or tmp not in vehicles_ids:
                tmp = tmp_dict[j]
                j += 1

            to_append = tmp
            l.append(to_append)
            vehicles_ids.remove(to_append)

    return l


def get_nonrandom_ids_small_groups(d, correct_vehicles_ids, num_groups, users_per_group):
    list_of_list = []
    vehicles_ids = [int(x) for x in correct_vehicles_ids]
    vehicles_ids.remove(2595)
    #vehicles_ids.remove(2580)

    for g in range(num_groups):
        l = []
        to_append = random.sample(vehicles_ids, 1)[0]
        l.append(int(to_append))
        vehicles_ids.remove(to_append)
        stop = False
        for i in range(users_per_group - 1):
            j = 0
            tmp_list = d[str(int(to_append))]
            # print(to_append)
            tmp = tmp_list[j]
            while tmp in l or tmp not in vehicles_ids:
                j += 1
                if j == len(tmp_list):
                    #print("STOP")
                    stop = True
                    break
                else:
                    tmp = tmp_list[j]

            if stop != True:
                to_append = tmp
                l.append(to_append)
                vehicles_ids.remove(to_append)
            else:
                break
        #if (len(l)) != 10:
        #    print(len(l))
        list_of_list.append(l)

    return list_of_list