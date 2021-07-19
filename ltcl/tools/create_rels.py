"""Creates a stational causal graph between balls"""

from physics_engine import BallEngine
import numpy as np
import ipdb as pdb

if __name__ == '__main__':
    dt = 0.02
    state_dim = 4
    n_ball = 5
    engine = BallEngine(dt, state_dim, action_dim=2)
    engine.init(n_ball)
    np.save('/data/datasets/logs/cmu_wyao/data/rels.npy', engine.param)
    print(engine.param)