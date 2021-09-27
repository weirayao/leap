"running the baseline file: main.py"
import warnings
warnings.filterwarnings('ignore')

import argparse
import os, pwd, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

"utils file (SAME)"
from ltcl.tools.utils import load_yaml
# Stationary: 
from ltcl.datasets.sim_dataset import SimulationDatasetTSTwoSample 
# Nonstationary: 
from ltcl.datasets.sim_dataset import SimulationDatasetTSTwoSampleNS

"baseline list"
from ltcl.baselines.TCL.model import TCL
# from ltcl.baselines.PCL.model import PCL # deprecated
from ltcl.baselines.iVAE.model import iVAE
from ltcl.baselines.BetaVAE.model import BetaVAE
from ltcl.baselines.SlowVAE.model import SlowVAE
from ltcl.baselines.FactorVAE.model import FactorVAE

def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../ltcl/configs', '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    pl.seed_everything(args.seed)

    if cfg['NS']:
      data = SimulationDatasetTSTwoSampleNS(directory=cfg['ROOT'], 
                                          transition=cfg['DATASET'])
    else:
      data = SimulationDatasetTSTwoSample(directory=cfg['ROOT'], 
                                          transition=cfg['DATASET'])

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

    if cfg['MODEL'] == "TCL":
      model = TCL(input_dim=cfg['VAE']['INPUT_DIM'],
                  z_dim=cfg['VAE']['LATENT_DIM'], 
                  nclass=cfg['TCL']['NCLASS'], 
                  hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                  lr=cfg['TCL']['LR'],
                  correlation=cfg['MCC']['CORR'])

    elif cfg['MODEL'] == "iVAE":
      model = iVAE(input_dim=cfg['VAE']['INPUT_DIM'],
                   z_dim=cfg['VAE']['LATENT_DIM'], 
                   hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                   lr=cfg['iVAE']['LR'],
                   correlation=cfg['MCC']['CORR'])

    elif cfg['MODEL'] == "BetaVAE":
      model = BetaVAE(input_dim=cfg['VAE']['INPUT_DIM'],
                      z_dim=cfg['VAE']['LATENT_DIM'], 
                      hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                      beta=cfg['BetaVAE']['BETA'], 
                      beta1=cfg['SlowVAE']['beta1_VAE'],
                      beta2=cfg['SlowVAE']['beta2_VAE'],
                      lr=cfg['BetaVAE']['LR'],
                      correlation=cfg['MCC']['CORR'])

    elif cfg['MODEL'] == "SlowVAE":
      model = SlowVAE(input_dim=cfg['VAE']['INPUT_DIM'],
                      z_dim=cfg['VAE']['LATENT_DIM'], 
                      hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                      beta=cfg['SlowVAE']['BETA'], 
                      gamma=cfg['SlowVAE']['GAMMA'], 
                      beta1=cfg['SlowVAE']['beta1_VAE'],
                      beta2=cfg['SlowVAE']['beta2_VAE'],
                      lr=cfg['SlowVAE']['LR'],
                      rate_prior=cfg['SlowVAE']['RATE_PRIOR'], 
                      correlation=cfg['MCC']['CORR'])

    elif cfg['MODEL'] == "FactorVAE":
      model = FactorVAE(input_dim=cfg['VAE']['INPUT_DIM'],
                        z_dim=cfg['VAE']['LATENT_DIM'], 
                        hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                        gamma=cfg['FactorVAE']['GAMMA'],
                        lr_VAE=cfg['FactorVAE']['LR_VAE'],
                        beta1_VAE=cfg['FactorVAE']['beta1_VAE'],
                        beta2_VAE=cfg['FactorVAE']['beta2_VAE'],
                        lr_D=cfg['FactorVAE']['LR_D'],
                        beta1_D=cfg['FactorVAE']['beta1_D'],
                        beta2_D=cfg['FactorVAE']['beta2_D'],
                        correlation=cfg['MCC']['CORR'])

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    trainer = pl.Trainer(default_root_dir=log_dir,
                         gpus=cfg['VAE']['GPU'], 
                         val_check_interval = cfg['MCC']['FREQ'],
                         max_epochs=cfg['VAE']['EPOCHS'],
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