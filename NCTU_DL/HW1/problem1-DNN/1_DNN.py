import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

from DNN_layer import RandomInit_Layer, ZeroInit_Layer, CrossEntropy, gradient_CrossEntropy, \
    error_rate

BATCH_SIZE = 60
NUM_EPOCH = 100

# LAYER_1 = 128
LAYER_1 = 256
LAYER_2 = 256
LAYER_3 = 128
LAYER_4 = 10

train_data = np.load('./train.npz')
train_image = train_data['image'].reshape(len(train_data['image']), 28*28)
train_label = train_data['label'].astype(int)

train_label_onehot = list()
for i in train_label:
    zero = np.zeros(10)
    zero[i] = 1
    train_label_onehot.append(zero)

test_data = np.load('./test.npz')
test_image = test_data['image'].reshape(len(test_data['image']), 28*28)
test_label = test_data['label'].astype(int)

test_label_onehot = list()
for i in test_label:
    zero = np.zeros(10)
    zero[i] = 1
    test_label_onehot.append(zero)

DnnLayer_1 = RandomInit_Layer(train_image.shape[1], LAYER_1)
DnnLayer_2 = RandomInit_Layer(LAYER_1, LAYER_2)
DnnLayer_3 = RandomInit_Layer(LAYER_2, LAYER_3)
DnnLayer_4 = RandomInit_Layer(LAYER_3, LAYER_4)

def forward(image_batch, layer_1=DnnLayer_1, layer_2=DnnLayer_2, 
        layer_3=DnnLayer_3, layer_4=DnnLayer_4):

    outputs = list()
    # Forward
    outputs.append(layer_1.forward(image_batch))
    outputs.append(layer_1.ReLU(outputs[-1]))

    outputs.append(layer_2.forward(outputs[-1]))
    outputs.append(layer_2.ReLU(outputs[-1]))

    outputs.append(layer_3.forward(outputs[-1]))
    outputs.append(layer_3.ReLU(outputs[-1]))

    outputs.append(layer_4.forward(outputs[-1]))

    return outputs

loss_train = []
loss_test = []

error_rate_train = []
error_rate_test = []

for epoch in range(1, NUM_EPOCH+1):
    for batch in range(0, train_image.shape[0], BATCH_SIZE):
        image_batch = train_image[batch: batch+BATCH_SIZE]
        label_batch = train_label[batch: batch+BATCH_SIZE]
        label_batch_onehot = train_label_onehot[batch: batch+BATCH_SIZE]

        outputs = forward(image_batch=image_batch)

        # Loss
        loss = CrossEntropy(outputs[-1], label_batch)
        loss_grad = gradient_CrossEntropy(outputs[-1], label_batch_onehot)

        # Backward
        loss_grad = DnnLayer_4.backward(outputs[-2], loss_grad)

        loss_grad = DnnLayer_3.ReLU_backward(outputs[-3], loss_grad)
        loss_grad = DnnLayer_3.backward(outputs[-4], loss_grad)

        loss_grad = DnnLayer_2.ReLU_backward(outputs[-5], loss_grad)
        loss_grad = DnnLayer_2.backward(outputs[-6], loss_grad)

        loss_grad = DnnLayer_1.ReLU_backward(outputs[-7], loss_grad)
        # loss_grad = DnnLayer_1.backward(outputs[-5], loss_grad)

    train_out = forward(image_batch=train_image)
    test_out = forward(image_batch=test_image)

    if epoch == 20:
        test_nodes2_20epoch = test_out[4]
    elif epoch == 80:
        test_nodes2_80epoch = test_out[4]

    loss_train.append(np.mean(CrossEntropy(train_out[-1], train_label)))
    loss_test.append(np.mean(CrossEntropy(test_out[-1], test_label)))

    error_rate_train.append(error_rate(train_out[-1], train_label_onehot))
    error_rate_test.append(error_rate(test_out[-1], test_label_onehot))

    print('Epoch: ', epoch)
    print('training loss: ', loss_train[-1])
    print('testing loss', loss_test[-1])
    print('training error rate: ', error_rate_train[-1])
    print('testing error rate: ', error_rate_test[-1], '\n')

plt.figure(1)
plt.title('Loss')
plt.xlabel('Num of Epoches')
plt.ylabel('Average Loss')
plt.plot(np.arange(0, NUM_EPOCH, 1), loss_train, color='r', label='training')
plt.plot(np.arange(0, NUM_EPOCH, 1), loss_test, color='b', label='testing')
plt.show()

plt.figure(2)
plt.title('Error Rate')
plt.xlabel('Num of Epoches')
plt.ylabel('Error Rate')
plt.plot(np.arange(0, NUM_EPOCH, 1), error_rate_train, color='r', label='training')
plt.plot(np.arange(0, NUM_EPOCH, 1), error_rate_test, color='b', label='testing')
plt.show()

plot_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
color = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'pink', 'gray', 'purple']

plt.figure(3)
plt.title('2D Features of 20th epoch')

l = []
for i in range(10):
    l.append([])
for i in range(len(test_nodes2_20epoch)):
    l[test_label[i]].append(test_nodes2_20epoch[i])

for i in range(9):
    tmp = np.array(l[i])
    plt.scatter(tmp[:, 0], tmp[:, 1], color=color[i], label=plot_label[i])
plt.show()

plt.figure(4)
plt.title('2D Features of 80th epoch')

l = []
for i in range(10):
    l.append([])
for i in range(len(test_nodes2_80epoch)):
    l[test_label[i]].append(test_nodes2_80epoch[i])

for i in range(9):
    tmp = np.array(l[i])
    plt.scatter(tmp[:, 0], tmp[:, 1], color=color[i], label=plot_label[i])
plt.show()

confusion_matrix = np.zeros((10, 10)).astype(int)
for o in range(len(test_out[-1])):
    predict = np.argmax(test_out[-1][o])
    real = test_label[o]
    confusion_matrix[predict][real] += 1

df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])

fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix)
ax.set_xticks(np.arange(len("0123456789")))
ax.set_yticks(np.arange(len("0123456789")))
ax.set_xticklabels("0123456789")
ax.set_yticklabels("0123456789")

for i in range(len("0123456789")):
    for j in range(len("0123456789")):
        text = ax.text(j, i, confusion_matrix[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Confusion Matrix")
fig.tight_layout()
plt.show()
