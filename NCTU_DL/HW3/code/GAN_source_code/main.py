from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from Model import *



# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def common_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', default='./data', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    
    return parser



def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs):
    generator.train()
    discriminator.train()

    d_loss_list = list()
    g_loss_list = list()

    sample_noise = torch.randn(64, nz, 1, 1, device=device)
    sample_list = list()

    # Each epoch, we have to go through every data in dataset
    for epoch in range(1, 1+num_epochs):
        print(f'{epoch} epoch')

        epoch_sample_list = list()

        # Each iteration, we will get a batch data for training
        for i, data in enumerate(dataloader, 0):

            # initialize gradient for network
            discriminator.zero_grad()
            # send the data into device for computation
            real_data = data[0].to(device)

            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data
            discriminator_real_output = discriminator.forward(real_data).view(-1)

            batch_size = real_data.size(0)
            label = torch.full((batch_size,), 1, device=device)

            discriminator_real_loss = criterion(discriminator_real_output, label)
            discriminator_real_loss.backward()
        
            ## Using Fake data, other steps are the same.
            # Generate a batch fake data by using generator
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_data = generator.forward(noise)
            label.fill_(0)

            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data
            discriminator_fake_output = discriminator.forward(fake_data.detach()).view(-1) # fake data No backpropagation

            # fake_label = torch.full((batch_size,), 0, device=device)
            discriminator_fake_loss = criterion(discriminator_fake_output, label)
            discriminator_fake_loss.backward()

            discriminator_loss = discriminator_real_loss + discriminator_fake_loss
            # Update your network
            optimizer_d.step()

            generator.zero_grad()
            output = discriminator.forward(fake_data).view(-1)
            label.fill_(1)

            generator_loss = criterion(output, label)
            generator_loss.backward()
            optimizer_g.step()
            
            # Record your loss every iteration for visualization
            d_loss_list.append(discriminator_loss.item())
            g_loss_list.append(generator_loss.item())
            
            # Use this function to output training procedure while training
            # You can also use this function to save models and samples after fixed number of iteration
            if i % 50 == 0:
                print(f'{i} iters, Discriminator Loss: {discriminator_loss.item()}, Generator Loss: {generator_loss.item()}')
                # print('Discriminator Real Loss', discriminator_real_loss.item(), ' Discriminator Fake Loss', discriminator_fake_loss.item())
     
            # Remember to save all things you need after all batches finished!!!
        generator.save(path=f'./genetator_{epoch}epoch.pkl')
        discriminator.save(path=f'./discriminator_{epoch}epoch.pkl')

        with torch.no_grad():
            sample = generator(sample_noise).detach().cpu()
            epoch_sample_list.append(vutils.make_grid(sample, padding=2, normalize=True).numpy())
            sample_list.append(vutils.make_grid(sample, padding=2, normalize=True).numpy())
 
        np.save(f'./generated_sample_{epoch}epoch.npy', np.array(epoch_sample_list))

    np.save('./generator_loss.npy', np.array(g_loss_list))
    np.save('./discriminator_loss.npy', np.array(d_loss_list))
    np.save(f'./generated_sample.npy', np.array(sample_list))
        

def main(args):
    # Create the dataset by using ImageFolder(get extra point by using customized dataset)
    # remember to preprocess the image by using functions in pytorch
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                    transforms.Resize(args.image_size, interpolation=2),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=2)

    # for i, data in enumerate(dataloader, 0):
    #     data = vutils.make_grid(data[0], padding=2, normalize=True).numpy()
    #     # fig = plt.figure(figsize=(8, 8))
    #     plt.figure()
    #     plt.axis("off")
    #     plt.imshow(np.transpose(data, (1, 2, 0)), animated=True)
    #     plt.show()
    #     break

    # Create the generator and the discriminator()
    # Initialize them 
    # Send them to your device
    generator = Generator(ngpu=ngpu).to(device)
    discriminator = Discriminator(ngpu=ngpu).to(device)

    print(generator)
    print(discriminator)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Setup optimizers for both G and D and setup criterion at the same time
    optimizer_g = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    
    # Start training~~
    train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, args.num_epochs)
    


if __name__ == '__main__':
    parser = common_arg_parser()
    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    main(args)