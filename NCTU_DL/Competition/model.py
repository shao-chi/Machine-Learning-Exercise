import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CNN(nn.Module):
    def __init__(self, settings):
        super(CNN, self).__init__()
        """
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        torch.nn.Linear(in_features, out_features, bias=True)

        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
            return_indices=False, ceil_mode=False)
        """
        # self.embedding = nn.Embedding(
        #     num_embeddings=settings['embedding']['num_embeddings'],
        #     embedding_dim=settings['embedding']['embedding_dim'],
        #     padding_idx=0)

        # self.pool = nn.MaxPool2d(
        #     kernel_size=settings['max_pool']['kernel_size'], 
        #     stride=settings['max_pool']['stride'])

        # self.conv1 = nn.Conv2d(
        #     in_channels=settings['cnn_layer_1']['in_channels'],
        #     out_channels=settings['cnn_layer_1']['out_channels'],
        #     kernel_size=settings['cnn_layer_1']['kernel_size'],
        #     stride=settings['cnn_layer_1']['stride'])
        # self.bn_1 = nn.BatchNorm2d(settings['cnn_layer_1']['out_channels'])

        # self.conv2 = nn.Conv2d(
        #     in_channels=settings['cnn_layer_2']['in_channels'],
        #     out_channels=settings['cnn_layer_2']['out_channels'],
        #     kernel_size=settings['cnn_layer_2']['kernel_size'],
        #     stride=settings['cnn_layer_2']['stride'])
        # self.bn_2 = nn.BatchNorm2d(settings['cnn_layer_2']['out_channels'])

        # self.conv3 = nn.Conv2d(
        #     in_channels=settings['cnn_layer_3']['in_channels'],
        #     out_channels=settings['cnn_layer_3']['out_channels'],
        #     kernel_size=settings['cnn_layer_3']['kernel_size'],
        #     stride=settings['cnn_layer_3']['stride'])
        # self.bn_3 = nn.BatchNorm2d(settings['cnn_layer_3']['out_channels'])

        self.fc1 = nn.Linear(
            in_features=settings['hidden_layer_1']['in_features'],
            out_features=settings['hidden_layer_1']['out_features'])
        self.fc2 = nn.Linear(
            in_features=settings['hidden_layer_2']['in_features'],
            out_features=settings['hidden_layer_2']['out_features'])
        self.fc3 = nn.Linear(
            in_features=settings['hidden_layer_3']['in_features'],
            out_features=settings['hidden_layer_3']['out_features'])

        self.bn_1 = nn.BatchNorm1d(settings['hidden_layer_1']['out_features'])
        self.bn_2 = nn.BatchNorm1d(settings['hidden_layer_2']['out_features'])

        self.softmax = nn.Softmax()

        self.settings = settings

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=settings['learning_rate'])

    def forward(self, x):
        # x = self.embedding(x).view(x.size(0), 1, x.size(1), self.settings['embedding']['embedding_dim'])

        # x = self.pool(F.relu(self.bn_1(self.conv1(x))))
        # x = F.relu(self.bn_2(self.conv2(x)))
        # x = F.relu(self.bn_3(self.conv3(x)))
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = self.softmax(self.fc3(x))

        return x

    def train_model(self, batch_x, batch_y):
        self.train()

        output = self.forward(batch_x)
        loss = self.criterion(output, batch_y)
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, x_data, y_target):
        with torch.no_grad():
            output = self.forward(x_data)

            return F.cross_entropy(output, y_target).item()

    def evaluate(self, x_data, y_target):
        with torch.no_grad():
            output = np.argmax(self.forward(x_data).numpy(), axis=1)
            # print(output[:20])
            # print(y_target[:20])

            # return sum(1 for x, y in zip(output, y_target.numpy()) if np.where(x == 1)[0] == y) / len(output)
            return sum(1 for x, y in zip(output, y_target.numpy()) if x == y) / len(output)

    def predict(self, x_data):
        with torch.no_grad():
            predictions = np.argmax(self.forward(x_data).numpy(), axis=1)

        return predictions.astype(int)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()


class LSTM(nn.Module):
    def __init__(self,
            hidden_size=512,
            num_layers=2,
            num_classes=43,
            learning_rate=0.0001,
            n_components=100):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(n_components,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=True)

        self.linear_2 = nn.Linear(hidden_size*2, num_classes)

        self.softmax = nn.Softmax()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, input_data):
        h0 = torch.zeros(self.num_layers*2, input_data.size(0), self.hidden_size).float()
        c0 = torch.zeros(self.num_layers*2, input_data.size(0), self.hidden_size).float()

        output, _ = self.lstm(input_data, (h0, c0))

        output = output[:, -1, :]
        # output = self.relu(self.linear_1(output))
        output = self.softmax(self.linear_2(output))

        return output


    def train_model(self, batch_x, batch_y):
        self.train()

        output = self.forward(batch_x)
        loss = self.criterion(output, batch_y)
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, x_data, y_target):
        with torch.no_grad():
            output = self.forward(x_data)

            return F.cross_entropy(output, y_target).item()

    def evaluate(self, x_data, y_target):
        with torch.no_grad():
            output = np.argmax(self.forward(x_data).numpy(), axis=1)
            # print(output[:20])
            # print(y_target[:20])

            # return sum(1 for x, y in zip(output, y_target.numpy()) if np.where(x == 1)[0] == y) / len(output)
            return sum(1 for x, y in zip(output, y_target.numpy()) if x == y) / len(output)

    def predict(self, x_data):
        with torch.no_grad():
            predictions = np.argmax(self.forward(x_data).numpy(), axis=1)

        return predictions.astype(int)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()