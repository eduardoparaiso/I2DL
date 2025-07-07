# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:50:23 2025

@author: Eduardo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Z_DIM = 256 
NUM_EPOCHS = 200
BATCH_SIZE = 256
LR_RATE = 1e-4


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 3,
                               out_channels= 32,
                               kernel_size= 4,
                               stride= 2,
                               padding= 1)
        
        self.conv2 = nn.Conv2d(in_channels= 32,
                               out_channels= 64,
                               kernel_size= 4,
                               stride= 2,
                               padding= 1)
        
        self.conv3 = nn.Conv2d(in_channels= 64,
                               out_channels= 128,
                               kernel_size= 4,
                               stride= 2,
                               padding= 1)
        
        self.conv4 = nn.Conv2d(in_channels= 128,
                               out_channels= 256,
                               kernel_size= 4,
                               stride= 2,
                               padding= 1)
        
        self.flattern = nn.Flatten()
        self.mu = nn.Linear(256*16*16, z_dim)
        self.sigma = nn.Linear(256*16*16, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #256 -> 128
        x = F.relu(self.conv2(x)) #128 -> 64
        x = F.relu(self.conv3(x)) #64 -> 32
        x = F.relu(self.conv4(x)) #32 -> 16

        x = self.flattern(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma
    
    
class Decoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256*16*16)
        self.deconv1 = nn.ConvTranspose2d(in_channels= 256,
                                          out_channels= 128,
                                          kernel_size= 4,
                                          stride= 2,
                                          padding= 1)
        
        self.deconv2 = nn.ConvTranspose2d(in_channels= 128,
                                          out_channels= 64,
                                          kernel_size= 4,
                                          stride= 2,
                                          padding= 1)
        
        self.deconv3 = nn.ConvTranspose2d(in_channels= 64,
                                          out_channels= 32,
                                          kernel_size= 4,
                                          stride= 2,
                                          padding= 1)
        
        self.deconv4 = nn.ConvTranspose2d(in_channels= 32,
                                          out_channels= 3,
                                          kernel_size= 4,
                                          stride= 2,
                                          padding= 1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 16, 16) #desfaz o flattern
        x = F.relu(self.deconv1(x)) # 16 -> 32
        x = F.relu(self.deconv2(x)) # 32 -> 64
        x = F.relu(self.deconv3(x)) # 64 -> 128
        x = torch.tanh(self.deconv4(x)) # 128 -> 256
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def reparameterize_trick(self, mu, sigma):
        std = torch.exp(sigma/2)
        epsilon = torch.randn_like(std)  #ruído gaussiano do mesmo shape do std
        return mu + epsilon * std

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize_trick(mu, sigma)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, sigma
    
    
def loss_function(x_reconstructed, x, mu, sigma):
    recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return recon_loss + kl_divergence, recon_loss, kl_divergence