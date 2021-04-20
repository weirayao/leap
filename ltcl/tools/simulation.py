"""Generate simulated data according to http://proceedings.mlr.press/v20/zhang11/zhang11.pdf"""
import numpy as np
import torch
import ltcl
from ltcl.baselines.GCL.mix import MixingMLP
from ltcl.modules.components.transforms import AfflineCoupling

def main():
    input_size = 4
    # Super-Gaussian is exp(Z) of standard normals
    # Sub-Gaussian is uniform distribution
    lags = 2
    input_size = 4
    latent_size = 4
    transitions = [ ]
    scale = 2
    for l in range(lags):
        B = (torch.rand(latent_size, latent_size) - 0.5)/scale
        scale = scale * 2
        transitions.append(B)
    # transitions[0] is B_{-L}, transotions[-1] is B_0/1
    transitions.reverse()

    mixing_func = AfflineCoupling(n_blocks = 3, 
                                  input_size = input_size, 
                                  hidden_size = 16, 
                                  n_hidden = 1, 
                                  batch_norm = False)
    length = 80 + lags # Use first lags elements as lags
    chunks = 500
    batch_size = 128
    for chunk_idx in range(chunks):
        batch_data = [ ]
        y = torch.rand(batch_size, lags, latent_size)
        for t in range(length):
            x1 = torch.exp(torch.normal(0, 0.5, size=(batch_size, latent_size//2)))
            x2 = torch.rand(batch_size, latent_size//2) - 0.5
            x = torch.cat((x1, x2), dim=1)
            for l in range(lags):
                x += torch.mm(y[:,l,:], transitions[l])
            x, _ = mixing_func.inverse(x)
            npx = x.detach().cpu().numpy()
            batch_data.append(npx)
            y = torch.cat((y[:,1:], x.unsqueeze(1)), dim=1)
        batch_data = np.stack(batch_data, axis=1)
        np.savez("/home/cmu_wyao/projects/data/%d"%chunk_idx, 
                 y=batch_data[:,:lags], x=batch_data[:,lags:])
    for l in range(lags):
        B = transitions[l].detach().cpu().numpy()
        np.save("/home/cmu_wyao/projects/data/W%d"%(lags-l), B)

if __name__ == "__main__":
    main()



