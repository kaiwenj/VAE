import sys
sys.paths.append('../src/')

from VAE import VAE, reparameterization, calculateLoss
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

## Adapted from https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f

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


class Train(object):

    def __init__(self, optimizer, device, batch_size, input_dim, epochs):
        self.optimizer=optimizer
        self.device=device
        self.batch_size=batch_size
        self.input_dim=input_dim
        self.epochs=epochs

    def __call__(self, model, train_loader):
        model.train()
        for epoch in range(self.epochs):
            overall_loss = 0
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(self.batch_size, self.input_dim).to(self.device)
                self.optimizer.zero_grad()
                loss = model(x)             
                overall_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*self.batch_size))
        return overall_loss


def main():

    # create a transform to apply to each datapoint
    transform = transforms.Compose([transforms.ToTensor()])
    
    # download the MNIST datasets
    path = '~/datasets'
    train_dataset = MNIST(path, transform=transform, download=True)
    test_dataset  = MNIST(path, transform=transform, download=True)
    
    # create train and test dataloaders
    batch_size = 100
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Training
    input_dim=784
    hidden_dim=400
    latent_dim=200

    encoder=Encoder(input_dim, hidden_dim, latent_dim)
    decoder=Decoder(input_dim, hidden_dim, latent_dim)

    model = VAE1(encoder, decoder, reparameterization, calculateLoss).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    training=Train(optimizer, device, 100, 784, 50)
    training(model, train_loader)

    ### Visualization
    def generate_digit(mean, var):
        z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
        x_decoded = model.decoder.decode(z_sample)
        digit = x_decoded.detach().cpu().reshape(28, 28) # reshape vector to 2d array
        plt.imshow(digit, cmap='gray')
        plt.axis('off')
        plt.show()
    
    generate_digit(0.0, 1.0), generate_digit(1.0, 0.0)

    def plot_latent_space(model, scale=1.0, n=50, digit_size=28, figsize=15):
        # display a n*n 2D manifold of digits
        figure = np.zeros((digit_size * n, digit_size * n))
    
        # construct a grid 
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]
    
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
                x_decoded = model.decoder.decode(z_sample)
                digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
                figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit
    
        plt.figure(figsize=(figsize, figsize))
        plt.title('VAE Latent Space Visualization')
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("mean, z [0]")
        plt.ylabel("var, z [1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.show()
    
    plot_latent_space(model)


if __name__=='__main__':
    main()


    
    

