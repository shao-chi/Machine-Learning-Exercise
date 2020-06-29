import cv2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

label = ['good', 'none', 'bad']

train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')

train_dict = train_csv.to_dict('records')
test_dict = test_csv.to_dict('records')

def data_prepare(data_dict):
    label = ['good', 'none', 'bad']

    train_data = []
    train_label = []
    for data in data_dict:
        filename = data['filename']

        img = cv2.imread(f'./images/{filename}')
        img = cv2.resize(img[data['ymin']: data['ymax'], data['xmin']: data['xmax']],
                 dsize=(80, 80), interpolation=cv2.INTER_CUBIC)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = np.array([img[:, :, i] for i in range(3)])
        train_data.append(img)

        for i in range(len(label)):
            if data['label'] == label[i]:
                train_label.append(i)
                
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    return train_data, train_label

train_data, train_label = data_prepare(train_dict)
test_data, test_label = data_prepare(test_dict)

LEARNING_RATE = 0.0005
BATCH_SIZE = 24
NUM_EPOCH = 20

LAYER_SETTINGS = {
    'cnn_layer_1': {
        'in_channels': 3,
        'out_channels': 32,
        'kernel_size': 3,
        'stride': 1
    },
    'cnn_layer_2': {
        'in_channels': 32,
        'out_channels': 64,
        'kernel_size': 3,
        'stride': 1
    },
    'cnn_layer_3': {
        'in_channels': 64,
        'out_channels': 128,
        'kernel_size': 3,
        'stride': 1
    },

    'max_pool': {
        'kernel_size': 2,
        'stride': 2
    },

    'hidden_layer_1': {
        'in_features': 128 * 8 * 8,
        'out_features': 128
    },
    'hidden_layer_2': {
        'in_features': 128,
        'out_features': 64
    },
    'hidden_layer_3': {
        'in_features': 64,
        'out_features': 3
    }
}

class CNN(nn.Module):
    def __init__(self, settings=LAYER_SETTINGS):
        super(CNN, self).__init__()
        """
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
            padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        torch.nn.Linear(in_features, out_features, bias=True)

        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, 
            return_indices=False, ceil_mode=False)
        """
        self.pool = nn.MaxPool2d(
            kernel_size=settings['max_pool']['kernel_size'], 
            stride=settings['max_pool']['stride'])

        self.conv1 = nn.Conv2d(
            in_channels=settings['cnn_layer_1']['in_channels'],
            out_channels=settings['cnn_layer_1']['out_channels'],
            kernel_size=settings['cnn_layer_1']['kernel_size'],
            stride=settings['cnn_layer_1']['stride'])
        self.conv2 = nn.Conv2d(
            in_channels=settings['cnn_layer_2']['in_channels'],
            out_channels=settings['cnn_layer_2']['out_channels'],
            kernel_size=settings['cnn_layer_2']['kernel_size'],
            stride=settings['cnn_layer_2']['stride'])
        self.conv3 = nn.Conv2d(
            in_channels=settings['cnn_layer_3']['in_channels'],
            out_channels=settings['cnn_layer_3']['out_channels'],
            kernel_size=settings['cnn_layer_3']['kernel_size'],
            stride=settings['cnn_layer_3']['stride'])

        self.fc1 = nn.Linear(
            in_features=settings['hidden_layer_1']['in_features'],
            out_features=settings['hidden_layer_1']['out_features'])
        self.fc2 = nn.Linear(
            in_features=settings['hidden_layer_2']['in_features'],
            out_features=settings['hidden_layer_2']['out_features'])
        self.fc3 = nn.Linear(
            in_features=settings['hidden_layer_3']['in_features'],
            out_features=settings['hidden_layer_3']['out_features'])

        self.settings = settings

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, self.settings['hidden_layer_1']['in_features'])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=0.9)

# train_data = np.load('train_data.npy') / 256
# train_label = np.load('train_label.npy')
# test_data = np.load('test_data.npy') / 256
# test_label = np.load('test_label.npy')
label_2 = np.load('label_2_data.npy') / 256

train_data = torch.tensor(train_data / 256).float()
train_label = torch.tensor(train_label)
test_data = torch.tensor(test_data / 256).float()
test_label = torch.tensor(test_label)
label_2 = torch.tensor(label_2).float()

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

for epoch in range(1, 1+NUM_EPOCH):

    for batch in range(0, len(train_data), BATCH_SIZE):
        batch_data = train_data[batch: batch+BATCH_SIZE]
        batch_label = train_label[batch: batch+BATCH_SIZE]

        seed = torch.randint(0, 103, (3,))
        tensor = torch.tensor([1, 1, 1])
        batch_data = torch.cat((batch_data, label_2[seed]), 0)
        batch_label = torch.cat((batch_label, tensor), 0)
        
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = cnn.forward(batch_data)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        # if batch % 36 == 0:
    with torch.no_grad():
        # evaluate the metrics of training data
        correct = 0

        outputs = cnn.forward(train_data)
        loss = F.cross_entropy(input=outputs, target=train_label).item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == train_label).sum().item()
        accuracy = correct / len(train_data)

        train_loss.append(loss)
        train_accuracy.append(accuracy)
        print('Epoch: ', epoch)
        print('Training...')
        print('Accuracy: ', accuracy, 'Loss: ', loss)

        # evaluate the metrics of testing data
        correct = 0

        outputs = cnn.forward(test_data)
        loss = F.cross_entropy(input=outputs, target=test_label).item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == test_label).sum().item()
        accuracy = correct / len(test_data)

        test_loss.append(loss)
        test_accuracy.append(accuracy)
        print('Testing...')
        print('Accuracy: ', accuracy, 'Loss: ', loss, '\n')
        
print('Finished Training')


np.save('train_loss_2.npy', np.array(train_loss))
np.save('test_loss_2.npy', np.array(test_loss))
np.save('train_accuracy_2.npy', np.array(train_accuracy))
np.save('test_accuracy_2.npy', np.array(test_accuracy))

with torch.no_grad():
    # evaluate the metrics of training data
    class_correct = list(0. for i in range(len(label)))
    class_total = list(0. for i in range(len(label)))

    outputs = cnn.forward(train_data)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == train_label).squeeze()
    for i in range(len(train_label)):
        l = train_label[i]
        class_correct[l] += c[i].item()
        class_total[l] += 1

    print('\nTrain...')
    for i in range(len(label)):
        print('Accuray of ', label[i], ': ', class_correct[i] / class_total[i])

    # evaluate the metrics of testing data
    class_correct = list(0. for i in range(len(label)))
    class_total = list(0. for i in range(len(label)))

    outputs = cnn.forward(test_data)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == test_label).squeeze()
    for i in range(len(test_label)):
        l = test_label[i]
        class_correct[l] += c[i].item()
        class_total[l] += 1

    print('\nTest...')
    for i in range(len(label)):
        print('Accuracy of ', label[i], ': ', class_correct[i] / class_total[i])

plt.figure(1)
plt.title('Loss')
plt.xlabel('Num of Epoches')
plt.ylabel('Loss')
plt.plot(np.arange(0, NUM_EPOCH, 1), train_loss, color='r', label='training')
plt.plot(np.arange(0, NUM_EPOCH, 1), test_loss, color='b', label='testing')
plt.legend()
plt.show()

plt.figure(2)
plt.title('Accuracy')
plt.xlabel('Num of Epoches')
plt.ylabel('Accuracy')
plt.plot(np.arange(0, NUM_EPOCH, 1), train_accuracy, color='r', label='training')
plt.plot(np.arange(0, NUM_EPOCH, 1), test_accuracy, color='b', label='testing')
plt.legend()
plt.show()
