import torch
import torch.nn as nn


class User(object):
    def __init__(self, dataloader, id, criterion, local_epochs, learning_rate):
        self.id = id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.local_epochs = local_epochs
        self.dataloader = dataloader
        self.learning_rate = learning_rate

    def update_weights(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        local_loss = 0
        local_total = 0
        criterion = nn.MSELoss(reduction="sum")

        for _ in range(self.local_epochs):

            for sample in self.dataloader:
                seq = sample["seq"].float().to(self.device)
                target = sample["target"].float().to(self.device)
                fixed = sample["fixed"].float().to(self.device)

                model.zero_grad()

                target_pred = model(seq, fixed)

                loss = criterion(target_pred, target)
                loss.backward()
                optimizer.step()

                local_loss += loss.item()
                local_total += target.nelement()


        return model.state_dict(), local_loss, local_total, len(self.dataloader.dataset)
