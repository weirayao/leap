import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from leap.modules.linear_vae import AfflineVAECNN
from leap.datasets.kitti import KittiMasks
from leap.tools.utils import load_yaml, setup_seed
from pytorch_lightning.callbacks import Callback

from train_spline import pretrain_spline
import os, pwd, yaml


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

    data = KittiMasks(path = os.path.join(cfg['ROOT'], cfg['DATASET']), 
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

    model = AfflineVAECNN(nc=cfg['VAE']['NC'],
                          z_dim=cfg['VAE']['LATENT_DIM'], 
                          lag=cfg['VAE']['LAG'],
                          hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                          bound=cfg['SPLINE']['BOUND'],
                          count_bins=cfg['SPLINE']['BINS'],
                          order=cfg['SPLINE']['ORDER'],
                          beta=cfg['VAE']['BETA'],
                          gamma=cfg['VAE']['GAMMA'],
                          l1=cfg['VAE']['L1'],
                          lr=cfg['VAE']['LR'],
                          diagonal=cfg['VAE']['DIAG'],
                          identity=cfg['VAE']['IDENTITY'], # Identity transitions
                          bias=cfg['VAE']['BIAS'],
                          use_warm_start=cfg['SPLINE']['USE_WARM_START'],
                          spline_pth=cfg['SPLINE']['PATH'],
                          decoder_dist=cfg['VAE']['DEC']['DIST'],
                          correlation=cfg['MCC']['CORR'])

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    trainer = pl.Trainer(default_root_dir=log_dir,
                         gpus=cfg['VAE']['GPU'], 
                         val_check_interval = cfg['MCC']['FREQ'],
                         max_epochs=cfg['VAE']['EPOCHS'])

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
    torch.cuda.empty_cache()
    main(args)