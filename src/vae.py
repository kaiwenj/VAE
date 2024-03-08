
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

def reparameterization(mean, var):
    epsilon = torch.randn_like(var)#.to(device)
    z = mean + var*epsilon
    return z

def calculateLoss(x, xHat, z_mu, log_z_sigma2):
    reproductionLoss = nn.functional.binary_cross_entropy(xHat, x, reduction='sum')
    dkl=-0.5*torch.sum(1+log_z_sigma2-z_mu.pow(2)-log_z_sigma2.exp())
    return reproductionLoss+dkl
