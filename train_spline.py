'''Pretrain Spline flow to a good intitialization point'''
import os
import argparse
import torch
import torch.nn.functional as F
from ltcl.tools.utils import load_yaml
from ltcl.modules.components.transforms import ComponentWiseSpline

def pretrain_spline(cfg_name):
    cfg = load_yaml(os.path.join('./ltcl/configs', 
                                 '%s.yaml'%cfg_name))
    batch_size = cfg['SPLINE']['BS']
    latent_size = cfg['SPLINE']['LATENT_DIM']
    bound = cfg['VAE']['LATENT_DIM']
    use_cuda = cfg['SPLINE']['CUDA']
    device = torch.device("cuda:0" if use_cuda else "cpu")
    steps = cfg['SPLINE']['STEPS']
    # Initialize the flow model
    flow  = ComponentWiseSpline(input_dim=latent_size,
                                bound=cfg['SPLINE']['BOUND'],
                                count_bins=cfg['SPLINE']['BINS'],
                                order=cfg['SPLINE']['ORDER'])
    flow.to(device)
    
    spline_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, flow.parameters()), 
                                        lr=cfg['SPLINE']['LR'],
                                        weight_decay=0.0)

    # Warm-start the prior to standard normal dist/laplacian dist
    for step in range(steps):
        if cfg['SPLINE']['TYPE'] == 'gaussian':
            y_t = torch.normal(0, 1, size=(batch_size, latent_size))
        elif cfg['SPLINE']['TYPE'] == 'laplacian':
            y_t = torch.distributions.laplace.Laplace(0, 1).rsample((batch_size, latent_size))
        dataset = y_t.to(device)
        spline_optimizer.zero_grad()
        z, logabsdet = flow(dataset)
        logp = flow.base_dist.log_prob(z) + logabsdet
        loss = -torch.mean(logp)
        loss.backward(retain_graph=True)
        spline_optimizer.step()

    # This checkpoint will be loaded in linear_vae.py
    torch.save(flow.state_dict(), cfg['SPLINE']['PATH'])

if __name__ ==  '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    args = argparser.parse_args()
    pretrain_spline(args.exp)