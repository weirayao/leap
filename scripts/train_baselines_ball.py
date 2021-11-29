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

from train_spline import pretrain_spline
from leap.tools.utils import load_yaml
from leap.datasets.physics_dataset import PhysicsDatasetTwoSample
from leap.modules.components.base import Namespace
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import ipdb as pdb

def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../leap/configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")
    data_cfg = load_yaml('../leap/configs/ball_5_s1.yaml')
    hparams = Namespace()
    for k in data_cfg:
        setattr(hparams, k, data_cfg[k])
    trans_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data = PhysicsDatasetTwoSample(hparams, trans_to_tensor=trans_to_tensor)
    pl.seed_everything(args.seed)

    # Warm-start spline
    if cfg['SPLINE']['USE_WARM_START']:
        if not os.path.exists(cfg['SPLINE']['PATH']):
            print('Pretraining Spline Flow...', end=' ', flush=True)
            pretrain_spline(args.exp)
            print('Done!')

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['VAE']['VAL_BS'], 
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            shuffle=False)

    if cfg['MODEL'] == "BetaVAE":
        from leap.baselines.BetaVAE.model import BetaBallKeypoint 
        model = BetaBallKeypoint(k=cfg['VAE']['K'],
                                 nc=cfg['VAE']['NC'],
                                 nf=cfg['VAE']['ENC']['NF'],
                                 beta=cfg['BetaVAE']['BETA'],
                                 lr=cfg['BetaVAE']['LR'],
                                 correlation=cfg['MCC']['CORR'],
                                 decoder_dist=cfg['VAE']['DEC']['DIST']) 

    elif cfg['MODEL'] == 'FactorVAE':
        from leap.baselines.FactorVAE.model import FactorBallKeypoint
        model = FactorBallKeypoint(k=cfg['VAE']['K'],
                                   nc=cfg['VAE']['NC'],
                                   nf=cfg['VAE']['ENC']['NF'],
                                   gamma=cfg['FactorVAE']['GAMMA'],
                                   hidden_dim=cfg['FactorVAE']['HIDDEN'],
                                   lr_VAE=cfg['FactorVAE']['LR_VAE'],
                                   lr_D=cfg['FactorVAE']['LR_D'], 
                                   correlation=cfg['MCC']['CORR'],
                                   decoder_dist=cfg['VAE']['DEC']['DIST']) 


    checkpoint_callback = ModelCheckpoint(monitor='val_mcc', 
                                          save_top_k=1, 
                                          mode='max')

    early_stop_callback = EarlyStopping(monitor="val_mcc", 
                                        min_delta=0.00, 
                                        patience=50, 
                                        verbose=False, 
                                        mode="max")
                                        
    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    trainer = pl.Trainer(default_root_dir=log_dir,
                         gpus=cfg['VAE']['GPU'], 
                         val_check_interval = cfg['MCC']['FREQ'],
                         max_epochs=cfg['VAE']['EPOCHS'],
                         deterministic=True,
                         callbacks=[checkpoint_callback])

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
