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
from leap.modules.srnn_cnn_kitti import SRNNConv
from leap.tools.utils import load_yaml
from leap.datasets.kitti import KittiMasksTwoSample
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


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

    pl.seed_everything(args.seed)
    
    # Warm-start spline
    if cfg['SPLINE']['USE_WARM_START']:
        if not os.path.exists(cfg['SPLINE']['PATH']):
            print('Pretraining Spline Flow...', end=' ', flush=True)
            pretrain_spline(args.exp)
            print('Done!')

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

    model = SRNNConv(nc=cfg['VAE']['NC'],
                     length=cfg['VAE']['LENGTH'],
                     z_dim=cfg['VAE']['LATENT_DIM'], 
                     z_dim_trans=cfg['VAE']['CAUSAL_DIM'], 
                     lag=cfg['VAE']['LAG'],
                     hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                     trans_prior=cfg['VAE']['TRANS_PRIOR'],
                     infer_mode=cfg['VAE']['INFER_MODE'],
                     bound=cfg['SPLINE']['BOUND'],
                     count_bins=cfg['SPLINE']['BINS'],
                     order=cfg['SPLINE']['ORDER'],
                     lr=cfg['VAE']['LR'],
                     l1=cfg['VAE']['L1'],
                     beta=cfg['VAE']['BETA'],
                     gamma=cfg['VAE']['GAMMA'],
                     sigma=cfg['VAE']['SIGMA'],
                     bias=cfg['VAE']['BIAS'],
                     use_warm_start=cfg['SPLINE']['USE_WARM_START'],
                     spline_pth=cfg['SPLINE']['PATH'],
                     decoder_dist=cfg['VAE']['DEC']['DIST'],
                     correlation=cfg['MCC']['CORR'])

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    checkpoint_callback = ModelCheckpoint(monitor='val_mcc', 
                                          save_top_k=1, 
                                          mode='max')

    # early_stop_callback = EarlyStopping(monitor="val_mcc", 
    #                                     min_delta=0.00, 
    #                                     patience=50, 
    #                                     verbose=False, 
    #                                     mode="max")
                                        
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
