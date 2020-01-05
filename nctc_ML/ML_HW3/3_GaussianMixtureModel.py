from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import numpy as np 
from PIL import Image

def Kmeans(data, K, iter):
    h, w, d = data.shape
    data = np.reshape(data, (-1, d))

    mu = data[np.random.choice(len(data), K, replace=False)]
    gamma_nk = np.ones([len(data), K])

    for i in range(iter):
        distort = np.sum((data[:, None] - mu) ** 2, axis=2) 
        new_gamma_nk = np.identity(K)[np.argmin(distort, axis=1)] 
        if new_gamma_nk.all() == gamma_nk.all():
            break
        else: 
            gamma_nk = new_gamma_nk
            
        mu = np.sum(gamma_nk[:, :, None] * (data[:, None]), axis=0) / np.sum(gamma_nk, axis=0)[:, None]

    print('K means (k = {})'.format(K))
    print('RGB:')
    for rgb in mu * 255:
        rgb = rgb.astype(int)
        print('[{}, {}, {}]'.format(rgb[0], rgb[1], rgb[2]))

    img = (mu[np.where(gamma_nk == 1)[1]] * 255).astype(int).reshape(h, w, d)
    plt.title('K means (k = {})'.format(K))
    plt.imshow(img)
    plt.show()

    return gamma_nk, mu

def GaussianMixture(data, K, Kmean_gamma_nk, Kmean_mu, iter):
    h, w, d = data.shape
    data = np.reshape(data, (-1, d))

    pi = np.sum(Kmean_gamma_nk, axis=0) / len(Kmean_gamma_nk)
    cov, gaussian = [], []
    for k in range(K):
        cov.append(np.cov(data[np.where(Kmean_gamma_nk[:, k] == 1)[0]].T))
        try:
            gaussian.append(multivariate_normal.pdf(data, mean=Kmean_mu[k], cov=cov[k]) * pi[k])
        except:
            Kmean_mu[k] = np.random.rand(d)
            cov[k] = np.random.rand(d, d).dot(np.random.rand(d, d).T)
            gaussian.append(multivariate_normal.pdf(data, mean=Kmean_mu[k], cov=cov[k]) * pi[k])

    cov = np.array(cov)
    gaussian = np.array(gaussian)
    likeli = []
    for i in range(iter):
        # E
        gamma = (gaussian / np.sum(gaussian, axis=0)).T

        # M
        N_k = np.sum(gamma, axis=0)
        mu = np.sum(gamma[:, :, None] * data[:, None], axis=0) / N_k[:, None]
        for k in range(K):
            cov[k] = (gamma[:, k, None] * (data - mu[k])).T.dot(data - mu[k]) / N_k[k] + 1e-7 * np.identity(d)

        pi = N_k / len(data)
        for k in range(K):
            try: 
                gaussian[k] = multivariate_normal.pdf(data, mean=mu[k], cov=cov[k]) * pi[k]
            except:
                mu[k] = np.random.rand(d)
                cov[k] = np.random.rand(d, d).dot(np.random.rand(d, d).T)
                gaussian[k] = multivariate_normal.pdf(data, mean=mu[k], cov=cov[k]) * pi[k]

        log_likelihood = np.sum(np.log(np.sum(gaussian, axis=0)))
        likeli.append(log_likelihood)

    print('Gaussian Mixture Model (k = {})'.format(K))
    print('RGB:')
    for rgb in mu * 255:
        rgb = rgb.astype(int)
        print('[{}, {}, {}]'.format(rgb[0], rgb[1], rgb[2]))

    plt.title('Log likelihood of Gaussian Mixture Model (k = {})'.format(K))
    plt.plot(np.linspace(1, 100, 100), likeli)
    plt.show()

    img = (mu[np.argmax(gaussian, axis=0)] * 255).astype(int).reshape(h, w, d)
    plt.title('Gaussian Mixture Model (k = {})'.format(K))
    plt.imshow(img)
    plt.show()

img = Image.open('./hw3_3.jpeg')
# img = Image.open('./IMG_6716.PNG')
data = np.array(img) / 255

n_iter = 100
K = [3, 5, 7, 10]
for k in K:
    gamma_nk, mu = Kmeans(data, k, n_iter)
    GaussianMixture(data, k, gamma_nk, mu, n_iter)