import argparse
import torchvision.transforms as transforms

import os
import numpy as np
from ltcl.datasets.physics_dataset import PhysicsDataset
from ltcl.tools.utils import load_yaml
import yaml
import ipdb as pdb

class Namespace(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    def __repr__(self):
        items = list(self.__dict__.items())
        temp = []
        for name, value in items:
            if not name.startswith('_'):
                temp.append('%s=%r' % (name, value))
        temp.sort()
        return '%s(%s)' % (self.__class__.__name__, ', '.join(temp))

def main(args):
    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"

    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.abspath(os.path.join(script_dir, rel_path))

    cfg = load_yaml(abs_file_path)

    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")
    # Genenrate args
    namespace = Namespace()
    for k in cfg:
        setattr(namespace, k, cfg[k])

    trans_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    sparsity = 0.67
    n_ball = cfg['n_ball']
    param_load = None
    if not cfg['variable_rels']:
        param_load = np.zeros((n_ball * (n_ball - 1) // 2, 2))
        n_rels = len(param_load)
        num_nonzero = int(n_rels * sparsity)
        choice = np.random.choice(n_rels, size=num_nonzero, replace=False)
        param_load[choice, 0] = 1
        param_load[choice, 1] = np.random.rand(num_nonzero) * 10

    datasets = {}
    # modMat = np.random.uniform(0, 1, (cfg['n_ball'], 2, cfg['n_class']))
    modMat = np.ones((cfg['n_ball'], 2, cfg['n_class']))
    for phase in range(cfg['n_class']):
        datasets[phase] = PhysicsDataset(namespace, phase=str(phase), trans_to_tensor=trans_to_tensor)
        datasets[phase].gen_data(modVec=modMat[:,:,phase], param_load=param_load)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    args = argparser.parse_args()
    main(args)



