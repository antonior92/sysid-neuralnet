

import torch
import torch.nn as nn






class LSTM(nn.Module):
    def __init__(self,  input_dim, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size)

        self.decoding_layers = nn.Conv1d(hidden_size, 1, 1)

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size),
                torch.zeros(1, batch_size, self.lstm.hidden_size))

    def forward(self, input_data: torch.Tensor):
        input_t = input_data.permute((2, 0, 1))

        out_t, _ = self.lstm(input_t, self.init_hidden(input_data.size(0)))

        out = out_t.permute(1, 2, 0)

        pred = self.decoding_layers(out)

        return pred
