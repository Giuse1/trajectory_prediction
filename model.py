import torch
import torch.nn as nn
torch.manual_seed(1)


class LSTMnn(nn.Module):

    def __init__(self, num_feat, hidden_dim, fixed_dim):
        super(LSTMnn, self).__init__()
        self.lstm = nn.LSTM(num_feat, hidden_dim, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128+fixed_dim, 2)

    def forward(self, seq, fixed):
        # input shape: (batch, seq_len, input_size)
        out, _ = self.lstm(seq)
        out = out[:, -11:-1, :]
        out = self.dense1(out)
        out = self.dense2(out)
        out = torch.cat((fixed.unsqueeze(1).repeat(1, out.shape[1], 1), out), 2)
        out = self.dense3(out)
        return out
