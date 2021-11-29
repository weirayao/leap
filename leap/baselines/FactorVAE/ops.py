"""ops.py"""

import torch
import torch.nn.functional as F


def recon_loss(x, x_recon, decoder_dist='gaussian'):
    n = x.size(0)
    if decoder_dist == 'gaussian':
        loss = F.mse_loss(x_recon, x, size_average=False).div(n)
    elif decoder_dist == 'bernoulli':
        loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss

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

    return total_kld

# def kl_divergence(mu, logvar):
#     kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
#     return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)
