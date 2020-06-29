import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygal.maps.world import World
from pygal.maps.world import COUNTRIES

from problem1_model import RNN, LSTM, GRU


data = pd.read_csv('covid_19.csv')
region = data.values[2:, 0]
count = data.values[2:, 3:].astype(float)


diff_count= np.zeros((count.shape[0], count.shape[1]-1))
label = np.zeros((count.shape[0], count.shape[1]-1))
for i in range(len(region)):
    for j in range(1, len(count[i])):
        diff_count[i][j-1] = count[i][j] - count[i][j-1]

        if diff_count[i][j-1] > 0:
            label[i][j-1] = 1.0


correlation = pd.DataFrame(data=diff_count.T, columns=region).astype('float32').corr()

# seed = np.random.randint(185, size=30)
# plot_corr = correlation.iloc[seed, seed]
# upper_triangle = np.tril(np.ones(correlation.iloc[seed, seed].shape)).astype(bool)
# plot_corr = plot_corr.where(upper_triangle)
# sns.heatmap(plot_corr, annot=False)
# plt.show()

target = 2
corr_list = correlation.values[target]
candidates = [target]
for i in range(1, len(corr_list)):
    if corr_list[i] > 0.6:
        candidates.append(i)

# candidates = np.arange(0, len(region), 1)
print('Nums of Candidate: ', len(candidates))

for i in range(len(diff_count)):
    diff_count[i] = (diff_count[i] - np.mean(diff_count[i])) / np.std(diff_count[i])
data = diff_count[candidates][20:]
label = label[candidates][20:]

L = 30
final_data = list()
final_label = list()
for i in range(len(data)):
    sequence = data[i]
    start_index = 0
    while len(sequence) - start_index > L:
        final_data.append(sequence[start_index:start_index+L])
        final_label.append(label[i][start_index+L])
        start_index += L

final_data = torch.tensor(np.array(final_data).reshape((len(final_data), -1, 1))).float()
final_label = torch.tensor(final_label).long()
split_index = int(len(final_data) * 0.75)
train_data = final_data[:split_index]
train_label = final_label[:split_index]
test_data = final_data[split_index:]
test_label = final_label[split_index:]

batch_size = 20
num_epochs = 500


def model_accuracy(output, label):
    _, predicted = torch.max(output, 1)
    correct = 0
    correct += (predicted == label).sum().item()
    accuracy = correct / len(label)

    return accuracy

def plot_figure(title, train_y, test_y, label_x, label_y):
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.plot(np.arange(1, len(train_y)+1, 1), train_y, color='r', label='training')
    plt.plot(np.arange(1, len(test_y)+1, 1), test_y, color='b', label='testing')
    plt.legend()
    plt.show()
    
rnn_model = RNN(hidden_size=128, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.00001)

# plt.title('Model')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')

train_acc = list()
train_loss = list()
test_acc = list()
test_loss = list()
for epoch in range(num_epochs):
    for i in np.arange(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        batch_label = train_label[i:i+batch_size]

        output = rnn_model(batch_data)
        loss = criterion(output, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        output = rnn_model(train_data)
        accuracy = model_accuracy(output, train_label)
        loss = criterion(output, train_label).item()
        train_acc.append(accuracy)
        train_loss.append(loss)

        output = rnn_model(test_data)
        accuracy = model_accuracy(output, test_label)
        loss = criterion(output, test_label).item()
        test_acc.append(accuracy)
        test_loss.append(loss)

    print(f'Epoch: {epoch+1}')
    print(f'Train Accuracy: {train_acc[-1]}, Loss: {train_loss[-1]}')
    print(f'Test Accuracy: {test_acc[-1]}, Loss: {test_loss[-1]}')

# plt.plot(np.arange(1, len(test_loss)+1, 1), test_loss, color='r', label='RNN')

plot_figure(
    title=f'RNN, Accuracy, L = {L}, Normalization',
    train_y=train_acc, 
    test_y=test_acc, 
    label_x='Iteration', 
    label_y='Accuracy')
plot_figure(
    title=f'RNN, Loss, L = {L}, Normalization', 
    train_y=train_loss, 
    test_y=test_loss, 
    label_x='Iteration', 
    label_y='Loss')

np.save('RNN_train_acc.npy', np.array(train_acc))
np.save('RNN_test_acc.npy', np.array(test_acc))
np.save('RNN_train_loss.npy', np.array(train_loss))
np.save('RNN_test_loss.npy', np.array(test_loss))


lstm_model = LSTM(hidden_size=128, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.00001)

train_acc = list()
train_loss = list()
test_acc = list()
test_loss = list()
for epoch in range(num_epochs):
    for i in np.arange(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        batch_label = train_label[i:i+batch_size]

        output = lstm_model(batch_data)
        loss = criterion(output, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        output = lstm_model(train_data)
        accuracy = model_accuracy(output, train_label)
        loss = criterion(output, train_label).item()
        train_acc.append(accuracy)
        train_loss.append(loss)

        output = lstm_model(test_data)
        accuracy = model_accuracy(output, test_label)
        loss = criterion(output, test_label).item()
        test_acc.append(accuracy)
        test_loss.append(loss)

    print(f'Epoch: {epoch+1}')   
    print(f'Train Accuracy: {train_acc[-1]}, Loss: {train_loss[-1]}')
    print(f'Test Accuracy: {test_acc[-1]}, Loss: {test_loss[-1]}')

# plt.plot(np.arange(1, len(test_loss)+1, 1), test_loss, color='b', label='LSTM')

plot_figure(
    title=f'LSTM, Accuracy, L = {L}, Normalization',
    train_y=train_acc, 
    test_y=test_acc, 
    label_x='Iteration', 
    label_y='Accuracy')
plot_figure(
    title=f'LSTM, Loss, L = {L}, Normalization', 
    train_y=train_loss, 
    test_y=test_loss, 
    label_x='Iteration', 
    label_y='Loss')

np.save('LSTM_train_acc.npy', np.array(train_acc))
np.save('LSTM_test_acc.npy', np.array(test_acc))
np.save('LSTM_train_loss.npy', np.array(train_loss))
np.save('LSTM_test_loss.npy', np.array(test_loss))


gru_model = GRU(hidden_size=128, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.00001)

train_acc = list()
train_loss = list()
test_acc = list()
test_loss = list()
for epoch in range(num_epochs):
    for i in np.arange(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        batch_label = train_label[i:i+batch_size]

        output = gru_model(batch_data)
        loss = criterion(output, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        output = gru_model(train_data)
        accuracy = model_accuracy(output, train_label)
        loss = criterion(output, train_label).item()
        train_acc.append(accuracy)
        train_loss.append(loss)

        output = gru_model(test_data)
        accuracy = model_accuracy(output, test_label)
        loss = criterion(output, test_label).item()
        test_acc.append(accuracy)
        test_loss.append(loss)

    print(f'Epoch: {epoch+1}')
    print(f'Train Accuracy: {train_acc[-1]}, Loss: {train_loss[-1]}')
    print(f'Test Accuracy: {test_acc[-1]}, Loss: {test_loss[-1]}')

# plt.plot(np.arange(1, len(test_loss)+1, 1), test_loss, color='g', label='GRU')
# plt.legend()
# plt.show()

plot_figure(
    title=f'GRU, Accuracy, L = {L}, Normalization',
    train_y=train_acc, 
    test_y=test_acc, 
    label_x='Iteration', 
    label_y='Accuracy')
plot_figure(
    title=f'GRU, Loss, L = {L}, Normalization', 
    train_y=train_loss, 
    test_y=test_loss, 
    label_x='Iteration', 
    label_y='Loss')

np.save('GRU_train_acc.npy', np.array(train_acc))
np.save('GRU_test_acc.npy', np.array(test_acc))
np.save('GRU_train_loss.npy', np.array(train_loss))
np.save('GRU_test_loss.npy', np.array(test_loss))

rnn_loss = np.load('RNN_test_loss.npy')
lstm_loss = np.load('LSTM_test_loss.npy')
gru_loss = np.load('GRU_test_loss.npy')
plt.title('Model')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(np.arange(1, len(rnn_loss)+1, 1), rnn_loss, color='r', label='RNN')
plt.plot(np.arange(1, len(lstm_loss)+1, 1), lstm_loss, color='b', label='LSTM')
plt.plot(np.arange(1, len(gru_loss)+1, 1), gru_loss, color='g', label='GRU')
plt.legend()
plt.show()

train = np.load('LSTM_train_acc.npy')
test = np.load('LSTM_test_acc.npy')
plt.title('LSTM, Acc, L = 30, Normalization')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.plot(np.arange(1, len(train)+1, 1), train, color='r', label='training')
plt.plot(np.arange(1, len(test)+1, 1), test, color='b', label='testing')
plt.legend()
plt.show()

all_data = diff_count[:, -L:]
all_data = all_data.reshape((len(all_data), -1, 1))
with torch.no_grad():
    output = rnn_model(torch.tensor(all_data).float())
    _, predicted = torch.max(output, 1)

changed_name = {
        'Bolivia': 'Bolivia, Plurinational State of',
        'Brunei': 'Brunei Darussalam',
        'Burma': 'Myanmar',
        'Cabo Verde': 'Cabo Verde',
        'Congo (Brazzaville)': 'Congo',
        'Congo (Kinshasa)': 'Congo, the Democratic Republic of the',
        'Czechia': 'Czech Republic',
        'Dominica': 'Dominican Republic',
        'Holy See': 'Holy See (Vatican City State)',
        'Iran': 'Iran, Islamic Republic of',
        'Korea, South': 'Korea, Republic of',
        'Libya': 'Libyan Arab Jamahiriya',
        'Moldova': 'Moldova, Republic of',
        'North Macedonia': 'Macedonia, the former Yugoslav Republic of',
        'Russia': 'Russian Federation',
        'South Sudan': 'Sudan',
        'Syria': 'Syrian Arab Republic',
        'Taiwan*': 'Taiwan, Province of China',
        'Tanzania': 'Tanzania, United Republic of',
        'US': 'United States',
        'Venezuela': 'Venezuela, Bolivarian Republic of',
        'Vietnam': 'Viet Nam'
}
country = {value:key for key, value in COUNTRIES.items()}
ascending = dict()
descending = dict()
for i in range(len(region)):
    if region[i] in changed_name.keys():
        region[i] = changed_name[region[i]]

    if region[i] in country.keys():
        code = country[region[i]]
    else:
        continue

    if predicted[i] == 1:
        ascending.update({code: output[i, predicted[i]]})
    else:
        descending.update({code: output[i, predicted[i]]})

print(ascending)
print(descending)

worldmap_chart = World()
worldmap_chart.title = 'Probability For each countries'
worldmap_chart.add('ascending', ascending)
worldmap_chart.add('descending', descending)
worldmap_chart.render_to_file('map5.svg')