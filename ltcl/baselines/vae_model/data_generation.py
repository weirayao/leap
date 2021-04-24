"""
Generate simulated data in both linear/nonlinear case

noise_term == nt
    # Super-Gaussian is exp(Z) of standard normals
    # Sub-Gaussian is uniform distribution
    
linear case:
y1t = y1t_1 - 1.5*y2t_1 + nt1
y2t = 1.2*y2t_1 + nt2
x1t = y1t + y2t
x2t = y1t - sin(y2t)

nonlinear case:
y1t = y1t_1 - np.square(y2t_1) + nt1
y2t = 1.2*y2t_1 + nt2
x1t = y1t + y2t
x2t = y1t - sin(y2t)

@author: 
"""
import os
import glob
import torch
import random
import numpy as np

DIR = "linear_data/" # "nonlinear_data/" in nonlinear case
VALIDATION_RATIO = 0.2

def main():
    lags = 1
    length = 50 + lags
    chunks = 200
    latent_size = 2
    batch_size = 64
    linear_type = True # False in nonlinear case
    
    for chunk_idx in range(chunks):
        batch_data = []
        # Initialize past latents
        y1t_1 = torch.rand(batch_size, 1)
        y2t_1 = torch.rand(batch_size, 1)
    
        if linear_type:
            for t in range(length):
                nt1 = torch.exp(torch.normal(0, 1, size=(batch_size, latent_size//2))) 
                nt2 = torch.rand(batch_size, latent_size//2) - 0.5 
                y1t = y1t_1 - 1.5*y2t_1 + nt1 
                y2t = 1.2*y2t_1 + nt2 
                x1t = y1t + y2t 
                x2t = y1t - torch.sin(y2t) 
                # save observed data
                xt =  torch.cat((x1t, x2t), dim=1) 
                npx = xt.detach().cpu().numpy()
                batch_data.append(npx)
                # step
                y1t_1 = y1t
                y2t_1 = y2t
        else: # nonlinear case
            for t in range(length):
                nt1 = torch.exp(torch.normal(0, 1, size=(batch_size, latent_size//2)))
                nt2 = torch.rand(batch_size, latent_size//2) - 0.5
                y1t = y1t_1 - np.square(y2t_1) + nt1
                y2t = 1.2*y2t_1 + nt2
                x1t = y1t + y2t
                x2t = y1t - torch.sin(y2t)
                # save observed data
                xt =  torch.cat((x1t, x2t), dim=1)
                npx = xt.detach().cpu().numpy()
                batch_data.append(npx)
                # step
                y1t_1 = y1t
                y2t_1 = y2t
    
        batch_data = np.stack(batch_data, axis=1) 
        np.savez("./linear_data/%d"%chunk_idx, xt_1=batch_data[:,:-1,:], xt=batch_data[:,1:,:])
        # "./nonlinear_data/" in nonlinear case

if __name__ == "__main__":
    main()
    datum_names = glob.glob(os.path.join(DIR, "*.npz"))
    n_samples = len(datum_names)
    # Shuffle samples
    random.shuffle(datum_names)
    n_train_samples = int((1-VALIDATION_RATIO)*n_samples)
    # Write training/val sample names to config files
    with open(os.path.join(DIR, "train.txt"), "w") as f:
        for datum_name in datum_names[:n_train_samples]:
            f.write('%s\n' % datum_name)
    with open(os.path.join(DIR, "val.txt"), "w") as f:
        for datum_name in datum_names[n_train_samples:]:
            f.write('%s\n' % datum_name)
