"""Generate simulated data according to http://proceedings.mlr.press/v20/zhang11/zhang11.pdf"""
import ltcl
import tqdm
import torch
import numpy as np
import torch.nn as nn

from ltcl.baselines.GCL.mix import MixingMLP
from ltcl.modules.components.transforms import AfflineCoupling

def ortho_init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)

def main():
    # Super-Gaussian is exp(Z) of standard normals
    # Sub-Gaussian is Laplace distribution
    lags = 2
    input_size = 8
    latent_size = 8
    transitions = [ ]
    scale = 2
    for l in range(lags):
        B = ((torch.rand(latent_size, latent_size) - 0.5)/scale).cuda()
        scale = scale * 2
        transitions.append(B)
    # transitions[0] is B_{-L}, transotions[-1] is B_0/1
    transitions.reverse()

    mixing_func = mixing_func = nn.Sequential(
                                                nn.Linear(latent_size, latent_size, bias=False),
                                                nn.LeakyReLU(0.2),
                                                nn.Linear(latent_size, latent_size, bias=False),
                                                nn.LeakyReLU(0.2),
                                                nn.Linear(latent_size, input_size, bias=False),
                                             ).cuda()
    mixing_func.apply(ortho_init_weights)

    # mixing_func = AfflineCoupling(n_blocks = 10,
    #                               input_size = input_size,
    #                               hidden_size = 256,
    #                               n_hidden = 8,
    #                               batch_norm = False).cuda()
    length = 80 + lags # Use first lags elements as lags
    chunks = 1000
    batch_size = 128
    for chunk_idx in tqdm.tqdm(range(chunks)):
        batch_data = [ ]
        # Initialize past latents
        y_l = torch.rand(batch_size, lags, latent_size).cuda() 
        for t in range(length):
            # Sample current noise y_t = [y_1, y_2]
            y_1 = torch.distributions.laplace.Laplace(0,0.1).rsample((batch_size, latent_size//2)).cuda()
            y_2 = torch.distributions.laplace.Laplace(0,0.1).rsample((batch_size, latent_size//2)).cuda()
            y_t = torch.cat((y_1, y_2), dim=1)
            for l in range(lags):
                y_t += torch.mm(y_l[:,l,:], transitions[l])
            x_t = mixing_func(y_t)
            npx = x_t.detach().cpu().numpy()
            batch_data.append(npx)
            # Update past latents
            y_l = torch.cat((y_l[:,1:], y_t.unsqueeze(1)), dim=1)
        # batch_data = [BS, length, input_size]
        batch_data = np.stack(batch_data, axis=1)
        np.savez("/home/cmu_wyao/projects/data/flow/%d"%chunk_idx, 
                 y=batch_data[:,:lags], x=batch_data[:,lags:])
    for l in range(lags):
        B = transitions[l].detach().cpu().numpy()
        np.save("/home/cmu_wyao/projects/data/flow/W%d"%(lags-l), B)

if __name__ == "__main__":
    main()



