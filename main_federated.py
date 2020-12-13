import torch
import torch.nn as nn
from model import LSTMnn
from FL.FL_train import train_model_aggregated, train_model, train_model_aggregated_small_groups
torch.manual_seed(1)


num_rounds = 100
local_epochs = 1
num_users = 100
users_per_group = 10
batch_size = 16
learning_rate = 1e-6
#path = "/content/drive/MyDrive/data_ngsim/"
#list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#total_num_users = len(list_files)
mode = "hybrid_random"


print(f"NUM_USERS: {num_users}")
print(f"users_per_group: {users_per_group}")
print(f"num_rounds: {num_rounds}")
print(f"local_epochs: {local_epochs}")
print(f"batch_size: {batch_size}")
print(f"learning_rate: {learning_rate}")
print(f"mode: {mode}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
model_ft = LSTMnn(num_feat=15, hidden_dim=128 , fixed_dim=3)
model_ft = model_ft.to(device)


if mode == "standard":
    train_loss, train_acc, val_loss, val_acc = train_model(model_ft, criterion, num_rounds=num_rounds, local_epochs=local_epochs, num_users=num_users,
                                                       batch_size=batch_size, learning_rate=learning_rate)

elif mode == "hybrid_random":
    train_loss, train_acc, val_loss, val_acc = train_model_aggregated(model_ft, criterion, num_rounds=num_rounds,
                                                           local_epochs=local_epochs,
                                                           num_users=num_users,
                                                           users_per_group=users_per_group, batch_size=batch_size,
                                                           learning_rate=learning_rate, mode=mode)

elif mode == "hybrid_non_random":
    train_loss, train_acc, val_loss, val_acc = train_model_aggregated(model_ft, criterion, num_rounds=num_rounds,
                                                           local_epochs=local_epochs,
                                                           num_users=num_users,
                                                           users_per_group=users_per_group, batch_size=batch_size,
                                                           learning_rate=learning_rate, mode=mode)

elif mode == "hybrid_non_random_small_groups":
    train_loss, train_acc, val_loss, val_acc = train_model_aggregated_small_groups(model_ft, criterion, num_rounds=num_rounds,
                                                           local_epochs=local_epochs,
                                                           num_users=num_users,
                                                           users_per_group=users_per_group, batch_size=batch_size,
                                                           learning_rate=learning_rate)
