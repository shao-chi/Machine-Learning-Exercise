import random
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec

from model import CNN, LSTM

LEARNING_RATE = 0.00005
BATCH_SIZE = 64
NUM_EPOCH = 50
NUM_CLASS = 10
EMBEDDING = 200

LAYER_SETTINGS = {
    'learning_rate': LEARNING_RATE,
    # 'embedding': {'num_embeddings':300000,
    #               'embedding_dim':EMBEDDING},
    # 'cnn_layer_1': {'in_channels': 1,
    #                 'out_channels': 32,
    #                 'kernel_size': 5,
    #                 'stride': 1},
    # 'cnn_layer_2': {'in_channels': 64,
    #                 'out_channels': 64,
    #                 'kernel_size': 2,
    #                 'stride': 1},
    # 'cnn_layer_3': {'in_channels': 64,
    #                 'out_channels': 128,
    #                 'kernel_size': 2,
    #                 'stride': 1},
    # 'max_pool': {'kernel_size': 2,
    #              'stride': 2},
    'hidden_layer_1': {'in_features': 11200,#64 * 14 * 1,
                       'out_features': 4096},
    'hidden_layer_2': {'in_features': 4096,
                       'out_features': 512},
    'hidden_layer_3': {'in_features': 512,
                       'out_features': NUM_CLASS}
}

def read_data(path, train=True):
    data = pd.read_csv(path)

    titles = data['title'].apply(literal_eval).values.tolist()
    keywords = data['keyword'].apply(literal_eval).values.tolist()
    # X = titles
    X = list()
    for t, k in zip(titles, keywords):
        X.append(t + k)

    if train:
        Y = data['label'].values.tolist()

        indices = np.arange(len(Y))
        np.random.shuffle(indices)
        X = [X[i] for i in indices]
        Y = [Y[i] for i in indices]

        # X = torch.LongTensor(X)
        Y = torch.LongTensor(Y)
        return X, Y

    else:
        idx = data['id'].values

        return idx, X


train_path = './data/results_v18/train_data_58187.csv'
test_path = './data/results_v18/test_data_58187.csv'

data_x, data_y = read_data(train_path)
test_id, test_x = read_data(test_path, False)

print('Training Word2Vec Model ...', end=' ')
word_model = word2vec.Word2Vec(sentences=data_x+test_x,
                               size=EMBEDDING,
                               min_count=1,
                               alpha=0.05,
                               negative=10,
                               window=2,
                               iter=5)
print('---Done---')

sequence_len = max(len(max(data_x, key=len)), len(max(test_x, key=len)))
print('Sequence Length: ', sequence_len)

data_x_vector = list()
for xx in data_x:
    vector = list()
    for i in range(len(xx)):
        vector.append(word_model.wv[xx[i]])

    if sequence_len > len(xx):
        zero = np.zeros(((sequence_len-len(xx)), EMBEDDING)).tolist()
        vector = zero + vector

    else:
        vector = vector[:sequence_len]

    # for i in range(sequence_len):
    #     if i >= len(xx):
    #         vector.append(np.zeros((EMBEDDING)))
    #     else:
    #         vector.append(word_model.wv[xx[i]])

    data_x_vector.append(vector)

data_x = torch.FloatTensor(data_x_vector)
del data_x_vector
data_x = data_x.view(data_x.size(0), 1, sequence_len, EMBEDDING)

test_x_vector = list()
for xx in test_x:
    vector = list()
    for i in range(len(xx)):
        vector.append(word_model.wv[xx[i]])

    if sequence_len > len(xx):
        zero = np.zeros(((sequence_len-len(xx)), EMBEDDING)).tolist()
        vector = zero + vector

    else:
        vector = vector[:sequence_len]

    # for i in range(sequence_len):
    #     if i >= len(xx):
    #         vector.append(np.zeros((EMBEDDING)))
    #     else:
    #         vector.append(word_model.wv[xx[i]])
    
    test_x_vector.append(vector)

test_x = torch.FloatTensor(test_x_vector)
del test_x_vector
test_x = test_x.view(test_x.size(0), 1, sequence_len, EMBEDDING)

torch.save(data_x, './data/results_v18/data_x_tensor_lin5.pt')
torch.save(data_y, './data/results_v18/data_y_tensor_lin5.pt')
torch.save(test_x, './data/results_v18/test_x_tensor_lin5.pt')
# data_x = torch.load('./data/results_v18/data_x_tensor_lin5.pt')
# data_y = torch.load('./data/results_v18/data_y_tensor_lin5.pt')
# test_x = torch.load('./data/results_v18/test_x_tensor_lin5.pt')

# data_x = data_x.view(data_x.size(0), data_x[0][0].size(0), EMBEDDING)
# test_x = test_x.view(test_x.size(0), test_x[0][0].size(0), EMBEDDING)

split = int(len(data_x)*0.95)
train_x = data_x[:split]
train_y = data_y[:split]
train_test_x = data_x[:10000]
train_test_y = data_y[:10000]
valid_x = data_x[split:]
valid_y = data_y[split:]

model = CNN(settings=LAYER_SETTINGS)

# model = LSTM(hidden_size=512,
#              num_layers=1,
#              num_classes=NUM_CLASS,
#              learning_rate=LEARNING_RATE,
#              n_components=EMBEDDING)

print('Training .......... ')
for epoch in range(1, 1+NUM_EPOCH):
    print(f'EPOCH: {epoch}')

    iteration = len(train_x) // BATCH_SIZE
    for iter_ in range(iteration):
        batch_x = train_x[iter_*BATCH_SIZE:(iter_+1)*BATCH_SIZE]
        batch_y = train_y[iter_*BATCH_SIZE:(iter_+1)*BATCH_SIZE]

        model.train_model(batch_x=batch_x, batch_y=batch_y)

        if (iter_) % 50 == 0:
            train_loss = model.compute_loss(x_data=train_test_x,
                                            y_target=train_test_y)
            train_acc = model.evaluate(x_data=train_test_x,
                                       y_target=train_test_y)
                                              
            valid_loss = model.compute_loss(x_data=valid_x,
                                            y_target=valid_y)
            valid_acc = model.evaluate(x_data=valid_x,
                                       y_target=valid_y)

            print(f'Iter {iter_+1}, TRAIN Loss: {round(train_loss, 4)}, Acc: {round(train_acc, 4)}, VALID Loss: {round(valid_loss, 4)}, Acc: {round(valid_acc, 4)}')

            if train_acc > 0.9:
                test_predictions = model.predict(x_data=test_x)
                test_predict = pd.DataFrame(data={'id': test_id, 'label': test_predictions})
                test_predict.to_csv(f'./test{int(train_acc*10000)}_v18.csv', index=False)
