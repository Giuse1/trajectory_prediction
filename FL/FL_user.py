import torch
import matplotlib.pyplot as plt
import numpy as np

class User(object):
    def __init__(self, dataloader, id, criterion, local_epochs, learning_rate):
        self.id = id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.local_epochs = local_epochs
        self.dataloader = dataloader
        self.learning_rate = learning_rate
        # self.height
        # self.width
        # #self.type
        # #self.direction
        # self.target_height
        # self.target_width
        # self.surrounding_widths
        # self.surrounding_heights
        # # self.surrounding_types

    def update_weights(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)

        for _ in range(self.local_epochs):

            local_loss = 0.0
            for epoch in range(1000):
                for sample in self.dataloader:
                    seq = sample["seq"].float().to(self.device)
                    target = sample["target"].float().to(self.device)
                    fixed = sample["fixed"].float().to(self.device)

                    model.zero_grad()

                    target_pred = model(seq, fixed)

                    loss = self.criterion(torch.squeeze(target_pred), target)
                    loss.backward()
                    optimizer.step()

                    local_loss += loss.item()

        return model.state_dict(), local_loss, len(self.dataloader.dataset)
