import torch
import torch.nn as nn
from .base import DynamicModule


class LSTM(DynamicModule):
    def __init__(self, num_inputs, num_outputs, hidden_size, num_layers=1, dropout=0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(num_inputs, hidden_size, num_layers, dropout=dropout)

        self.decoding_layers = nn.Conv1d(hidden_size, num_outputs, 1)
        self.has_internal_state = True
        self.requested_output = 'same'

    def set_requested_output(self, requested_output):
        self.requested_output = requested_output

    def get_requested_output(self):
        return self.requested_output

    def get_requested_input(self, requested_output='internal'):
        if requested_output == 'internal':
            requested_output = self.requested_output
        return requested_output

    def init_hidden(self, batch_size, device=None):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device))

    def forward(self, input_data, state):
        input_t = input_data.permute((2, 0, 1))

        out_t, next_state = self.lstm(input_t, state)

        out = out_t.permute(1, 2, 0)

        pred = self.decoding_layers(out)

        return pred, next_state
