import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        self.encode_1 = nn.Linear(image_size, hidden_size)
        self.encode_2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_size) # mean
        self.log_var = nn.Linear(hidden_size, latent_size) # std

        self.decode_1 = nn.Linear(latent_size, hidden_size)
        self.decode_2 = nn.Linear(hidden_size, image_size)

        # self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, inputs):
        outputs = self.encode_1(inputs)
        outputs = self.relu(outputs)
        outputs = self.encode_2(outputs)
        outputs = self.relu(outputs)

        return outputs

    def gaussian_param_projection(self, inputs):
        mean = self.mean(inputs)
        log_var = self.log_var(inputs)

        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        if self.training:
            std = torch.exp(log_var/2)
            eps = torch.randn_like(std)

            return mean + eps * std
        else: 
            return mean

    def decode(self, z):
        outputs = self.decode_1(z)
        outputs = self.relu(outputs)
        outputs = self.decode_2(outputs)
        outputs = self.relu(outputs)
        
        outputs = self.sigmoid(outputs)

        return outputs
    
    def forward(self, inputs):
        outputs = self.encode(inputs)
        mean, log_var = self.gaussian_param_projection(outputs)
        z = self.reparameterize(mean, log_var)
        inputs_reconstruction = self.decode(z)

        return inputs_reconstruction, mean, log_var