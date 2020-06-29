import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(1, hidden_size, num_layers, batch_first = True, dropout = 0.5)
        self.linear = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax()

    def forward(self, input_data):
        h0 = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).float()

        output, _ = self.rnn(input_data, h0)

        output = output[:, -1, :]
        output = self.linear(output)
        output = self.softmax(output)

        return output


class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first = True, dropout = 0.5)
        self.linear = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax()

    def forward(self, input_data):
        h0 = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).float()
        c0 = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).float()

        output, _ = self.lstm(input_data, (h0, c0))

        output = output[:, -1, :]
        output = self.linear(output)
        output = self.softmax(output)

        return output


class GRU(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(1, hidden_size, num_layers, batch_first = True, dropout = 0.5)
        self.linear = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax()

    def forward(self, input_data):
        h0 = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).float()

        output, _ = self.gru(input_data, h0)

        output = output[:, -1, :]
        output = self.linear(output)
        output = self.softmax(output)

        return output