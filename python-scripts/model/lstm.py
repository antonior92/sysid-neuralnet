import torch
import torch.nn as nn
from .base import DynamicModule


class LSTM(DynamicModule):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(num_inputs, hidden_size)

        self.decoding_layers = nn.Conv1d(hidden_size, num_outputs, 1)
        self.has_internal_state = True

    def requested_input(self, requested_output):
        return requested_output

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size),
                torch.zeros(1, batch_size, self.lstm.hidden_size))

    def forward(self, input_data, state):
        input_t = input_data.permute((2, 0, 1))

        out_t, next_state = self.lstm(input_t, state)

        out = out_t.permute(1, 2, 0)

        pred = self.decoding_layers(out)

        return pred, next_state
