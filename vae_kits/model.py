import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

def l2(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return (2 - 2 * (x * y).sum(dim=-1))

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


# loss functions
def reconstruction_loss(x, x_recon, distribution='gaussian'):
    
    batch_size = x.size(0)  # [256 B, 163]
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'weighted_bernoulli':
         weight = torch.tensor([0.1, 0.9]).to("cuda") # just a label here
         weight_ = torch.ones(x.shape).to("cuda")
         weight_[x <= 0.5] = weight[0]
         weight_[x > 0.5] = weight[1]
         recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none')
         recon_loss = torch.sum(weight_ * recon_loss).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'poisson':
        # print((x - x_recon * torch.log(x)).shape)
        x_recon.clamp(min=1e-7, max=1e7)
        recon_loss = torch.sum(x_recon - x * torch.log(x_recon)).div(batch_size)
    elif distribution == 'poisson2':
        # layer = nn.Softplus()
        # x_recon = layer(x_recon)
        x_recon = x_recon + 1e-7
        recon_loss = torch.sum(x_recon - x * torch.log(x_recon)).div(batch_size)
    else:
        raise NotImplementedError

    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# model
class VAE_neural(nn.Module):
    """linear VAE developed to train neural dataset."""
    input_size = 163
    def __init__(self, l_dim=20, hidden_dim = [163, 128], batchnorm=False): # nc actually indicate if it is RGB image
        super(VAE_neural, self).__init__()

        self.l_dim = l_dim
        self.layers_dim = [self.input_size, *hidden_dim] # [163, (163, 128)]

        e_modules = []
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            e_modules.append(nn.Linear(in_dim, out_dim))
            if batchnorm:
                e_modules.append(nn.BatchNorm1d(num_features=out_dim))
            e_modules.append(nn.ReLU(True))
        e_modules.append(nn.Linear(self.layers_dim[-1], 2*self.l_dim))
        self.encoder = nn.Sequential(*e_modules)

        self.layers_dim.reverse()
        d_modules = []
        d_modules.append(nn.Linear(self.l_dim, self.layers_dim[0]))
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            if batchnorm:
                d_modules.append(nn.BatchNorm1d(num_features=in_dim))
            d_modules.append(nn.ReLU(True))
            d_modules.append(nn.Linear(in_dim, out_dim))
        self.decoder = nn.Sequential(*d_modules)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        # print('x', x)
        distributions = self._encode(x)
        mu = distributions[:, :self.l_dim]
        logvar = distributions[:, self.l_dim:]
        z = reparametrize(mu, logvar)

        x_recon = (self._decode(z).view(x.size()))
        recon_loss = reconstruction_loss(x, x_recon, distribution='bernoulli')
        kl_loss, _, _ = kl_divergence(mu, logvar)

        return recon_loss, kl_loss

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def _representation(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.l_dim]
        return mu

class swapVAE_neural(nn.Module):
    """ swap VAE developed to train neural dataset.
        part of the latent representation is used for clustering, the rest is used to VAE
    """
    def __init__(self, s_dim=64, l_dim=128, input_size=163, hidden_dim = [163, 128], batchnorm=True):
        super(swapVAE_neural, self).__init__()

        self.input_size = input_size # number of neurons

        self.s_dim = s_dim
        self.l_dim = l_dim
        self.c_dim = int(l_dim - s_dim)
        self.layers_dim = [self.input_size, *hidden_dim] # [163, (163, 128)]

        e_modules = []
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            e_modules.append(nn.Linear(in_dim, out_dim))
            if batchnorm:
                e_modules.append(nn.BatchNorm1d(num_features=out_dim))
            e_modules.append(nn.ReLU(True))
        e_modules.append(nn.Linear(self.layers_dim[-1], self.l_dim + self.s_dim))
        self.encoder = nn.Sequential(*e_modules)

        self.layers_dim.reverse()
        d_modules = []
        d_modules.append(nn.Linear(self.l_dim, self.layers_dim[0]))
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            if batchnorm:
                d_modules.append(nn.BatchNorm1d(num_features=in_dim))
            d_modules.append(nn.ReLU(True))
            d_modules.append(nn.Linear(in_dim, out_dim))
        # test softplus here
        d_modules.append(nn.Softplus())
        self.decoder = nn.Sequential(*d_modules)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x1, x2):
        # get c and s for x1
        distributions1 = self._encode(x1)
        c1 = distributions1[:, :self.c_dim]
        mu1 = distributions1[:, self.c_dim:self.l_dim]
        logvar1 = distributions1[:, self.l_dim:]
        s1 = reparametrize(mu1, logvar1)

        # get c and s for x2
        distributions2 = self._encode(x2)
        c2 = distributions2[:, :self.c_dim]
        mu2 = distributions2[:, self.c_dim:self.l_dim]
        logvar2 = distributions2[:, self.l_dim:]
        s2 = reparametrize(mu2, logvar2)

        # create new z1 and z2 by exchanging the content
        z1_new = torch.cat([c2, s1], dim=1)
        z2_new = torch.cat([c1, s2], dim=1)

        #### exchange content reconsturction
        x1_recon = (self._decode(z1_new).view(x1.size()))
        x2_recon = (self._decode(z2_new).view(x1.size()))

        #### original reconstruction
        z1_ori = torch.cat([c1, s1], dim=1)
        z2_ori = torch.cat([c2, s2], dim=1)
        x1_recon_ori = (self._decode(z1_ori).view(x1.size()))
        x2_recon_ori = (self._decode(z2_ori).view(x1.size()))

        distribution_label = "poisson"

        recon1 = reconstruction_loss(x1, x1_recon, distribution=distribution_label)
        recon2 = reconstruction_loss(x2, x2_recon, distribution=distribution_label)
        recon1_ori = reconstruction_loss(x1, x1_recon_ori, distribution=distribution_label)
        recon2_ori = reconstruction_loss(x2, x2_recon_ori, distribution=distribution_label)
        kl1, _, _ = kl_divergence(mu1, logvar1)
        kl2, _, _ = kl_divergence(mu2, logvar2)

        l2_loss = l2(c1, c2).mean()

        images = torch.cat([x1, x1_recon, x2, x2_recon], dim=0)
        return l2_loss, (recon1 + recon2) / 2, (kl1 + kl2) / 2, (recon1_ori + recon2_ori) / 2, images

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def _representation(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.l_dim]
        return mu

    def _representation_c(self, x):
        distributions = self._encode(x)
        c = distributions[:, :self.c_dim]
        return c

    def _representation_s(self, x):
        distributions = self._encode(x)
        s = distributions[:, self.c_dim:self.l_dim]
        return s

    def _reconstruct(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.l_dim]
        recon = self._decode(mu).view(x.size())
        return recon

