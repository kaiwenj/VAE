
import torch
import torch.nn as nn

class VAE(nn.Module):

    def __init__(self, encoder, decoder, reparameterization, calculateLoss):
        super(VAE, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.reparameterization=reparameterization
        self.calculateLoss=calculateLoss

    def forward(self, x):
        z_mu, log_z_sigma2=self.encoder.encode(x)
        z=self.reparameterization(z_mu, log_z_sigma2)
        xHat=self.decoder.decode(z)
        loss=self.calculateLoss(x, xHat, z_mu, log_z_sigma2)
        return loss

def reparameterization(mu, sigma2):
    epsilon = torch.randn_like(sigma2)#.to(device)
    z = mu + sigma2 * epsilon
    return z

def calculateLoss(x, xHat, mu, log_sigma2):
    reproductionLoss = nn.functional.binary_cross_entropy(xHat, x, reduction='sum')
    dkl=-0.5*torch.sum(1+log_sigma2-mu.pow(2)-log_sigma2.exp())
    return reproductionLoss+dkl
