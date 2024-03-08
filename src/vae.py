
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
