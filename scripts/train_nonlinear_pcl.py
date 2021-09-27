import torch
import random
import argparse
import numpy as np
import os, pwd, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from ltcl.tools.utils import load_yaml, setup_seed
from train_spline import pretrain_spline
from ltcl.datasets.sim_dataset import SimulationDatasetPCLNS
from ltcl.baselines.PCL.model import PCL


def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../ltcl/configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    data = SimulationDatasetPCLNS(directory=cfg['ROOT'], 
                                  transition=cfg['DATASET'],
                                  lags=cfg['PCL']['LAG'])
    pl.seed_everything(args.seed)

    num_validation_samples = cfg['PCL']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])
    train_loader = DataLoader(train_data, 
                              batch_size=cfg['PCL']['TRAIN_BS'], 
                              pin_memory=cfg['PCL']['PIN'],
                              num_workers=cfg['PCL']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['PCL']['VAL_BS'], 
                            pin_memory=cfg['PCL']['PIN'],
                            num_workers=cfg['PCL']['CPU'],
                            shuffle=False)

    model = PCL(z_dim=cfg['PCL']['LATENT_DIM'], 
                lags=cfg['PCL']['LAG'], 
                hidden_dims=cfg['PCL']['HIDDEN_DIM'], 
                encoder_layers=cfg['PCL']['ENCODER_LAYER'], 
                scoring_layers=cfg['PCL']['SCORE_LAYER'],
                correlation=cfg['MCC']['CORR'],
                lr=cfg['PCL']['LR'])

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    trainer = pl.Trainer(default_root_dir=log_dir,
                         gpus=cfg['PCL']['GPU'], 
                         val_check_interval = cfg['MCC']['FREQ'],
                         max_epochs=cfg['PCL']['EPOCHS'])

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )
    args = argparser.parse_args()
    main(args)
