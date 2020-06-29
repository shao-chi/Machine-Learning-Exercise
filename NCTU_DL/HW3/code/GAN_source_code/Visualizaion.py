import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

##Write down your visualization code here

# Animation for your generation
#input : image_list (size = (the number of sample times, how many samples created each time, image )   )
for e in range(1, 6):
    img = np.load(f'./generated_sample_{e}epoch.npy')

    # fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(np.transpose(img[0], (1, 2, 0)), animated=True)
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    plt.show()
    # break
# # https://matplotlib.org/api/_as_gen/matplotlib.animation.Animation.html#matplotlib.animation.Animation.save
img = np.load(f'./generated_sample.npy')

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()

g_loss = np.load('./generator_loss.npy')
d_loss = np.load('./discriminator_loss.npy')

x = np.arange(1, len(g_loss)+1, 1)
plt.figure()
plt.title('Loss During Training')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(x, g_loss, label='Generator')
plt.plot(x, d_loss, label='Discriminator')
plt.legend()
plt.show()