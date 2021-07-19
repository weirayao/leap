import os
import yaml
import argparse
import ipdb as pdb
import torchvision.transforms as transforms

from ltcl.tools.utils import load_yaml
from ltcl.datasets.physics_dataset import PhysicsDataset


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

    datasets = {}
    phase = 'raw'
    datasets[phase] = PhysicsDataset(namespace, phase=phase, trans_to_tensor=trans_to_tensor)
    datasets[phase].gen_data()

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    args = argparser.parse_args()
    main(args)



