import torch
import random
import argparse
import numpy as np
import ipdb as pdb
import os, pwd, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import warnings
warnings.filterwarnings('ignore')

from ltcl.modules.keypointer import Keypointer
from ltcl.tools.utils import load_yaml
from ltcl.modules.components.base import Namespace
from ltcl.datasets.physics_dataset import PhysicsDataset
import torchvision.transforms as transforms


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
    data_cfg = load_yaml(cfg['DATA_CFG'])
    hparams = Namespace()
    for k in data_cfg:
        setattr(hparams, k, data_cfg[k])
    trans_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data = PhysicsDataset(hparams, trans_to_tensor=trans_to_tensor)
    pl.seed_everything(args.seed)

    num_validation_samples = cfg['KEYPOINTER']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, 
                              batch_size=cfg['KEYPOINTER']['TRAIN_BS'], 
                              pin_memory=cfg['KEYPOINTER']['PIN'],
                              num_workers=cfg['KEYPOINTER']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['KEYPOINTER']['VAL_BS'], 
                            pin_memory=cfg['KEYPOINTER']['PIN'],
                            num_workers=cfg['KEYPOINTER']['CPU'],
                            shuffle=False)

    model = Keypointer(n_kps=cfg['KEYPOINTER']['N_KPS'],
                       width=cfg['KEYPOINTER']['WIDTH'],
                       height=cfg['KEYPOINTER']['HEIGHT'], 
                       nf=cfg['KEYPOINTER']['NF'], 
                       norm_layer=cfg['KEYPOINTER']['NORM'], 
                       lim=cfg['KEYPOINTER']['LIM'],
                       lr=cfg['KEYPOINTER']['LR'])

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    trainer = pl.Trainer(default_root_dir=log_dir,
                         gpus=cfg['KEYPOINTER']['GPU'], 
                         val_check_interval = cfg['KEYPOINTER']['FREQ'],
                         max_epochs=cfg['KEYPOINTER']['EPOCHS'],
                         deterministic=True)

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
