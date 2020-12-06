import torch
import torch.nn as nn
from model import LSTMnn
from FL.FL_train import train_model_aggregated, train_model
import os
torch.manual_seed(1)

num_rounds = 150
local_epochs = 1
num_users = 100
# users_per_group = 10
batch_size = 8
learning_rate = 0.0001
#path = "/content/drive/MyDrive/data_ngsim/"
#list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#total_num_users = len(list_files)
mode = "standard"


print(f"NUM_USERS: {num_users}")
# print(f"users_per_group: {users_per_group}")
print(f"num_rounds: {num_rounds}")
print(f"local_epochs: {local_epochs}")
print(f"batch_size: {batch_size}")
print(f"learning_rate: {learning_rate}")
print(f"mode: {mode}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
model_ft = LSTMnn(num_feat=15, hidden_dim=128 , fixed_dim=3)
model_ft = model_ft.to(device)
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)


if mode == "standard":
    train_loss, train_acc, val_loss, val_acc = train_model(model_ft, criterion, num_rounds=num_rounds, local_epochs=local_epochs, num_users=num_users,
                                                       batch_size=batch_size, learning_rate=learning_rate, iid=True)

elif mode == "hybrid":
    train_loss, train_acc, val_loss, val_acc = train_model_aggregated(model_ft, criterion, num_rounds=num_rounds,
                                                           local_epochs=local_epochs, total_num_users=total_num_users,
                                                           num_users=num_users,
                                                           users_per_group=users_per_group, batch_size=batch_size,
                                                           learning_rate=learning_rate, iid=True)
