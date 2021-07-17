import argparse
import pdb
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pytorch_lightning as pl

from ltcl.modules.physics_vae import PhysicsVAE
from ltcl.modules.components.base import Namespace
from ltcl.datasets.physics_dataset import PhysicsDataset
from ltcl.tools.utils import load_yaml
from pytorch_lightning.callbacks import Callback

from train_spline import pretrain_spline
import os, pwd, yaml

def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"

    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('./ltcl/configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    # Converts dict to argparse hparams
    hparams = Namespace()
    for k in cfg:
        setattr(hparams, k, cfg[k])

    # Warm-start spline
    if cfg['SPLINE']['USE_WARM_START']:
        if not os.path.exists(cfg['SPLINE']['PATH']):
            print('Pretraining Spline Flow...', end=' ', flush=True)
            pretrain_spline(args.exp)
            print('Done!')

    trans_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.9920, 0.9887, 0.9860), (0.0692, 0.0670, 0.0949))
    ])

    data = PhysicsDataset(hparams, phase='raw', trans_to_tensor=trans_to_tensor)

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']

    train_data, val_data, _ = random_split(data, [int(1e5), num_validation_samples, len(data)-num_validation_samples-int(1e5)])

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

    model = PhysicsVAE(nc=cfg['VAE']['NC'],
                       n_obj=cfg['VAE']['N_OBJ'],
                       z_dim=cfg['VAE']['LATENT_DIM'], 
                       lag=cfg['VAE']['LAG'],
                       hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                       num_layers=cfg['VAE']['GNN']['NUM_LAYERS'],
                       layer_name=cfg['VAE']['GNN']['LAYER_NAME'],
                       bound=cfg['SPLINE']['BOUND'],
                       count_bins=cfg['SPLINE']['BINS'],
                       order=cfg['SPLINE']['ORDER'],
                       beta=cfg['VAE']['BETA'],
                       gamma=cfg['VAE']['GAMMA'],
                       l1=cfg['VAE']['L1'],
                       lr=cfg['VAE']['LR'],
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
    args = argparser.parse_args()
    main(args)
