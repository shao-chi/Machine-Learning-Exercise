import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from problem2_model import VAE

data = list()
for i in np.arange(1, 21552, 1):
    img = cv2.imread(f'./data/{i}.png')
    img = cv2.resize(img, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
    data.append(img.reshape((-1)))

data = torch.tensor(data).float()

# def MSE_loss(inputs_reconstruction, inputs):
#     return F.mse_loss(
#         input=inputs_reconstruction, 
#         target=inputs,
#         reduction='sum')

def KL_div(mean, log_var):
    return - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())


num_epoch = 300
batch_size = 50

hidden_size = 256
latent_size = 16

model = VAE(image_size=data.size(1), hidden_size=hidden_size, latent_size=latent_size)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss(reduction='sum')

ELBO = list()
for epoch in np.arange(1, num_epoch+1, 1):
    train_loss = 0
    train_bce = 0
    train_kl = 0
    print(f'Epoch: {epoch}')

    batch_index = batch_size
    while batch_index < len(data):
        batch_data = data[batch_index-batch_size:batch_index] / 255
        batch_index += batch_size

        reconstruction, mean, log_var = model.forward(batch_data)
        batch_BCE = criterion(reconstruction, batch_data)
        batch_KL = KL_div(
            mean=mean, 
            log_var=log_var).mul(100)
        loss = batch_BCE + batch_KL
        # print(batch_BCE.item(), batch_KL.item(), loss.item())
        # print(batch_KL.item(), batch_KL.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bce += batch_BCE.item()
        train_kl += batch_KL.item()

    # with torch.no_grad():
    #     reconstruction, mean, log_var = model.forward(data)
    #     BCE = BCE_loss(
    #         inputs_reconstruction=reconstruction,
    #         inputs=data)
    #     KL = KL_div(
    #         mean=mean, 
    #         log_var=log_var)
    #     loss = BCE + KL

    #     ELBO.append(loss.item() / len(data))

    ELBO.append(train_loss / len(data))
    print(f'Training ELBO: {ELBO[-1]}, BCE: {train_bce / len(data)}, KL: {train_kl / len(data)}')

torch.save(model.state_dict(), '256_16_model_100KL.pkl')

ELBO = np.array(ELBO)
np.save('256_16_model_ELBO_100KL.npy', ELBO)


model = VAE(image_size=data.size(1), hidden_size=hidden_size, latent_size=latent_size)
model.load_state_dict(torch.load('256_32_model_100KL.pkl'))
model.eval()

y = np.load('256_16_model_ELBO_100KL.npy')
x = np.arange(1, num_epoch, 1)
plt.title('VAE ELBO')
plt.plot(x, y[1:])
plt.xlabel('epoch')
plt.ylabel('ELBO')
plt.show()

with torch.no_grad():
    batch_data = list()
    for i in np.arange(1, batch_size+1, 1):
        img = cv2.imread(f'./data/{i}.png')
        batch_data.append(img)
    batch_data = np.array(batch_data)
    tmp_img = batch_data[0]
    for j in range(1, 5):
        img = batch_data[j]
        tmp_img = cv2.hconcat([tmp_img, img])
    final_img = tmp_img
    for i in np.arange(5, batch_size, 5):
        tmp_img = batch_data[i]
        for j in range(1, 5):
            img = batch_data[i+j]
            tmp_img = cv2.hconcat([tmp_img, img])
        final_img = cv2.vconcat([final_img, tmp_img])
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

with torch.no_grad():
    batch_data = data[0:batch_size] / 255
    out, _, _ = model.forward(batch_data)
    out = out.view(-1, 16, 16, 3).numpy()
    tmp_img = cv2.resize(out[0], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    for j in range(1, 5):
        img = cv2.resize(out[j], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        tmp_img = cv2.hconcat([tmp_img, img])
    final_img = tmp_img
    for i in np.arange(5, batch_size, 5):
        tmp_img = cv2.resize(out[i], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        for j in range(1, 5):
            img = cv2.resize(out[i+j], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            tmp_img = cv2.hconcat([tmp_img, img])
        final_img = cv2.vconcat([final_img, tmp_img])
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

with torch.no_grad():
    z = torch.randn(batch_size, latent_size)
    out = model.decode(z).view(-1, 16, 16, 3).numpy()
    tmp_img = cv2.resize(out[0], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    for j in range(1, 5):
        img = cv2.resize(out[j], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        tmp_img = cv2.hconcat([tmp_img, img])
    final_img = tmp_img
    for i in np.arange(5, batch_size, 5):
        tmp_img = cv2.resize(out[i], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        for j in range(1, 5):
            img = cv2.resize(out[i+j], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            tmp_img = cv2.hconcat([tmp_img, img])
        final_img = cv2.vconcat([final_img, tmp_img])
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

with torch.no_grad():
    z = data[8:10] / 255
    batch_data = list()
    for i in np.arange(0, 1.1, 0.1):
        tmp = z[0] + (z[1] - z[0]) * i
        batch_data.append(tmp.numpy())
    batch_data = torch.tensor(batch_data).float()
    out, _, _ = model.forward(batch_data)
    out = out.view(-1, 16, 16, 3).numpy()
    tmp_img = cv2.resize(out[0], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    for j in range(1, 11):
        img = cv2.resize(out[j], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        tmp_img = cv2.hconcat([tmp_img, img])
    plt.imshow(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
