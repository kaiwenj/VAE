import sys
sys.paths.append('../src/')

import VAE
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid




class Encoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

class Decoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
        
    def decode(self, x):
        x = self.decoder(x)
        return x


