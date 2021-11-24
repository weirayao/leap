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
from ltcl.tools.utils import load_yaml
from ltcl.datasets.kitti import KittiMasksTwoSample
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


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

    pl.seed_everything(args.seed)

    data = KittiMasksTwoSample(path = os.path.join(cfg['ROOT'], cfg['DATASET']), 
                               transform = cfg['TRANSFORM'],
                               max_delta_t = cfg['DT'])

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']

    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])
    # Disable augmentation in validation set
    val_data.dataset.transform = None

    train_loader = DataLoader(train_data, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=True,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['VAE']['VAL_BS'], 
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            shuffle=False)

    if cfg['MODEL'] == "BetaVAE":
        from ltcl.baselines.BetaVAE.model import BetaKittiConv
        model = BetaKittiConv(nc=cfg['VAE']['NC'],
                              z_dim=cfg['VAE']['LATENT_DIM'],
                              hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                              beta=cfg['BetaVAE']['BETA'],
                              lr=cfg['BetaVAE']['LR'],
                              correlation=cfg['MCC']['CORR'],
                              decoder_dist=cfg['VAE']['DEC']['DIST'])

    elif cfg['MODEL'] == "FactorVAE":
        from ltcl.baselines.FactorVAE.model import FactorVAEKitti
        model = FactorVAEKitti(nc=cfg['VAE']['NC'],
                               z_dim=cfg['VAE']['LATENT_DIM'],
                               hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                               gamma=cfg['FactorVAE']['GAMMA'],
                               lr_VAE=cfg['FactorVAE']['LR_VAE'],
                               lr_D=cfg['FactorVAE']['LR_D'], 
                               correlation=cfg['MCC']['CORR'],
                               decoder_dist=cfg['VAE']['DEC']['DIST'])

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    checkpoint_callback = ModelCheckpoint(monitor='val_mcc', 
                                          save_top_k=1, 
                                          mode='max')
                                        
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
