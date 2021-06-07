import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from ltcl.modules.linear_vae import AfflineVAESynthetic
from ltcl.datasets.sim_dataset import SimulationDataset
from ltcl.tools.utils import load_yaml

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

    # Warm-start spline
    if cfg['SPLINE']['USE_WARM_START']:
        if not os.path.exists(cfg['SPLINE']['PATH']):
            print('Pretraining Spline Flow...', end=' ', flush=True)
            pretrain_spline(args.exp)
            print('Done!')
        else:
            print('Load Spline Checkpoint', flush=True)

    data = SimulationDataset(directory=cfg['ROOT'], 
                             transition=cfg['DATASET'])

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']

    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['VAE']['VAL_BS'], 
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            shuffle=False)

    model = AfflineVAESynthetic(input_dim=cfg['VAE']['INPUT_DIM'],
                                z_dim=cfg['VAE']['LATENT_DIM'], 
                                lag=cfg['VAE']['LAG'],
                                hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                                beta=cfg['VAE']['BETA'],
                                gamma=cfg['VAE']['GAMMA'],
                                lr=cfg['VAE']['LR'],
                                diagonal=cfg['VAE']['DIAG'],
                                use_warm_start=cfg['SPLINE']['USE_WARM_START'],
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
