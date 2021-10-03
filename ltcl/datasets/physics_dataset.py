"""
Physics dataset class from:
https://raw.githubusercontent.com/pairlab/v-cdn/master/data.py

Causal Discovery in Physical Systems from Videos
https://arxiv.org/abs/2007.00631

"""
import multiprocessing as mp
import os
import time

from PIL import Image

import cv2
import numpy as np
import imageio
import pickle
import scipy.misc
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset

from ltcl.tools.physics_engine import BallEngine

from ltcl.tools.utils import rand_float, rand_int
from ltcl.tools.utils import init_stat, combine_stat, load_data, store_data
from ltcl.tools.utils import resize, crop
from ltcl.tools.utils import adjust_brightness, adjust_saturation, adjust_contrast, adjust_hue
import random
import ipdb as pdb

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def normalize(data, stat, var=False):
    for i in range(len(stat)):
        stat[i][stat[i][:, 1] == 0, 1] = 1.0
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = (data[i] - s[:, 0]) / s[:, 1]
    else:
        for i in range(len(stat)):
            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]
    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]
    return data


def get_crop_params(phase, img, crop_size):
    w, h = img.size

    if w < h:
        tw = crop_size
        th = int(crop_size * h / w)
    else:
        th = crop_size
        tw = int(crop_size * w / h)

    if phase == 'train':
        if w == tw and h == th:
            return 0, 0, h, w
        assert False
        i = rand_int(0, h - th)
        j = rand_int(0, w - tw)

    else:
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

    return i, j, th, tw


def resize_and_crop(phase, src, scale_size, crop_size):
    # resize the images
    src = resize(src, scale_size)

    # crop the images
    crop_params = get_crop_params(phase, src, crop_size)
    src = crop(src, crop_params[0], crop_params[1], crop_params[2], crop_params[3])

    return src


def default_loader(path):
    return pil_loader(path)


def gen_Ball(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    dt, video, args, phase = info['dt'], info['video'], info['args'], info['phase']
    n_ball = info['n_ball']
    modVec = info['mod_vec']
    param_load = info['param_load']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim    # radius
    state_dim = args.state_dim  # x, y, xdot, ydot
    action_dim = 2              # ddx, ddy

    stats = [init_stat(attr_dim), init_stat(state_dim), init_stat(action_dim)]

    engine = BallEngine(dt, state_dim, action_dim=2)

    for i in tqdm(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)
        # Use the same relations for every episode
        engine.init(n_ball, param_load=param_load)

        n_obj = engine.num_obj
        attrs_all = np.zeros((time_step, n_obj, attr_dim))
        states_all = np.zeros((time_step, n_obj, state_dim))
        actions_all = np.zeros((time_step, n_obj, action_dim))
        rel_attrs_all = np.zeros((time_step, engine.param_dim, 2))

        act = np.zeros((n_obj, 2))
        for j in range(time_step):
            state = engine.get_state()

            vel_dim = state_dim // 2
            pos = state[:, :vel_dim]
            vel = state[:, vel_dim:]

            if j > 0:
                vel = (pos - states_all[j - 1, :, :vel_dim]) / dt

            attrs = np.zeros((n_obj, attr_dim))
            attrs[:] = engine.radius

            attrs_all[j] = attrs
            states_all[j, :, :vel_dim] = pos
            states_all[j, :, vel_dim:] = vel
            rel_attrs_all[j] = engine.param
            act += (np.random.rand(n_obj, 2) - 0.5) * 600 - act * 0.1 - state[:, 2:] * 0.1
            act = np.clip(act, -1000, 1000)
            act = np.multiply(act, modVec)
            # act += np.random.laplace(loc=0.0, scale=0.5, size=(5,2)) * 300 - act * 0.1 - state[:, 2:] * 0.1
            # act += (np.random.rand(n_obj, 2) - 0.5) * 600 - act * 0.1 - state[:, 2:] * 0.1
            # act = np.clip(act, -1000, 1000)
            engine.step(act)

            actions_all[j] = act.copy()

        datas = [attrs_all, states_all, actions_all, rel_attrs_all]
        store_data(data_names, datas, rollout_dir + '.h5')
        engine.render(states_all, actions_all, engine.get_param(), video=False, image=True,
                      path=rollout_dir, draw_edge=False, verbose=False)

        datas = [datas[i].astype(np.float64) for i in range(len(datas))]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats

class PhysicsDataset(Dataset):

    def __init__(self, args, phase='raw', trans_to_tensor=None, loader=default_loader):

        self.args = args
        self.phase = phase
        self.trans_to_tensor = trans_to_tensor
        self.loader = loader

        dataf = os.path.join(args.root_dir, args.dataf + '_' + args.env)
        self.dataf = dataf
        self.data_dir = os.path.join(dataf, phase)
        self.stat_path = os.path.join(dataf, 'stat.h5')
        self.stat = None
        os.system('mkdir -p ' + self.data_dir)
        self.data_names = ['attrs', 'states', 'actions', 'rels']
        ratio = self.args.train_valid_ratio
        self.n_rollout = self.args.n_rollout
        self.T = self.args.time_step
        self.scale_size = args.scale_size
        self.crop_size = args.crop_size
        self.n_class = self.args.n_class
        self.length  = 6

    def load_data(self):
        self.stat = load_data(self.data_names, self.stat_path)

    def gen_data(self, modVec = None, param_load=None):
        # if the data hasn't been generated, generate the data
        n_rollout, time_step, dt = self.n_rollout, self.args.time_step, self.args.dt
        assert n_rollout % self.args.num_workers == 0

        print("Generating data ... n_rollout=%d, time_step=%d" % (n_rollout, time_step))
        infos = []
        for i in range(self.args.num_workers):
            info = {'thread_idx': i,
                    'data_dir': self.data_dir,
                    'nonstationary': self.args.variable_rels,
                    'data_names': self.data_names,
                    'n_rollout': n_rollout // self.args.num_workers,
                    'time_step': time_step,
                    'dt': dt,
                    'video': False,
                    'phase': self.phase,
                    'args': self.args,
                    'vis_height': self.args.height_raw,
                    'vis_width': self.args.width_raw, 
                    'mod_vec': modVec,
                    'param_load': param_load}

            info['env'] = 'Ball'
            info['n_ball'] = self.args.n_ball
                
            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)
        env = self.args.env
        data = pool.map(gen_Ball, infos)

        print("Training data generated, warpping up stats ...")

        self.stat = [init_stat(self.args.attr_dim),
                     init_stat(self.args.state_dim),
                     init_stat(self.args.action_dim)]

        for i in range(len(data)):
            for j in range(len(self.stat)):
                self.stat[j] = combine_stat(self.stat[j], data[i][j])

        store_data(self.data_names[:len(self.stat)], self.stat, self.stat_path)

    def __len__(self):
        args = self.args
        length = self.n_class * self.n_rollout * args.time_step
        return length

    def __getitem__(self, idx):
        args = self.args
        suffix = '.png'
        offset = args.time_step - self.length + 1
        src_phase = idx % self.n_class
        src_rollout = (idx // self.n_class) // args.time_step
        src_timestep = (idx // self.n_class) % args.time_step
        des_phase = rand_int(0, self.n_class)
        des_rollout = rand_int(0, self.n_rollout)
        des_timestep = rand_int(0, args.time_step)
        src_path = os.path.join(self.dataf, str(src_phase), str(src_rollout), 'fig_%d%s' % (src_timestep, suffix))
        des_path = os.path.join(self.dataf, str(des_phase), str(des_rollout), 'fig_%d%s' % (des_timestep, suffix))

        src = self.loader(src_path)
        des = self.loader(des_path)

        src = resize_and_crop(self.phase, src, self.scale_size, self.crop_size)
        des = resize_and_crop(self.phase, des, self.scale_size, self.crop_size)

        src = self.trans_to_tensor(src)
        des = self.trans_to_tensor(des)
        sample = {'src': src, 'des': des}
        return sample

class PhysicsDatasetTwoSample(Dataset):

    def __init__(self, args, phase='raw', trans_to_tensor=None, loader=default_loader):

        self.args = args
        self.phase = phase
        self.trans_to_tensor = trans_to_tensor
        self.loader = loader

        dataf = os.path.join(args.root_dir, args.dataf + '_' + args.env)
        self.dataf = dataf
        self.data_dir = os.path.join(dataf, phase)
        self.stat_path = os.path.join(dataf, 'stat.h5')
        self.stat = None
        os.system('mkdir -p ' + self.data_dir)
        self.data_names = ['attrs', 'states', 'actions', 'rels']
        ratio = self.args.train_valid_ratio
        self.n_rollout = self.args.n_rollout
        self.T = self.args.time_step
        self.scale_size = args.scale_size
        self.crop_size = args.crop_size
        self.n_class = self.args.n_class
        self.length  = 6

    def load_data(self):
        self.stat = load_data(self.data_names, self.stat_path)

    def gen_data(self, modVec = None, param_load=None):
        # if the data hasn't been generated, generate the data
        n_rollout, time_step, dt = self.n_rollout, self.args.time_step, self.args.dt
        assert n_rollout % self.args.num_workers == 0

        print("Generating data ... n_rollout=%d, time_step=%d" % (n_rollout, time_step))
        infos = []
        for i in range(self.args.num_workers):
            info = {'thread_idx': i,
                    'data_dir': self.data_dir,
                    'nonstationary': self.args.variable_rels,
                    'data_names': self.data_names,
                    'n_rollout': n_rollout // self.args.num_workers,
                    'time_step': time_step,
                    'dt': dt,
                    'video': False,
                    'phase': self.phase,
                    'args': self.args,
                    'vis_height': self.args.height_raw,
                    'vis_width': self.args.width_raw, 
                    'mod_vec': modVec,
                    'param_load': param_load}

            info['env'] = 'Ball'
            info['n_ball'] = self.args.n_ball
                
            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)
        env = self.args.env
        data = pool.map(gen_Ball, infos)

        print("Training data generated, warpping up stats ...")

        self.stat = [init_stat(self.args.attr_dim),
                     init_stat(self.args.state_dim),
                     init_stat(self.args.action_dim)]

        for i in range(len(data)):
            for j in range(len(self.stat)):
                self.stat[j] = combine_stat(self.stat[j], data[i][j])

        store_data(self.data_names[:len(self.stat)], self.stat, self.stat_path)

    def __len__(self):
        args = self.args
        length = self.n_class * self.n_rollout * (args.time_step - self.length + 1)
        return length
    
    def retrieve_by_index(self, idx):
        args = self.args
        suffix = '.png'
        offset = args.time_step - self.length + 1
        src_phase = idx % self.n_class
        src_rollout = (idx // self.n_class) // offset
        src_timestep = (idx // self.n_class) % offset
        '''
        used for dynamics modeling
        '''
        imgs = []
        arrs = []
        data_path = os.path.join(self.dataf, str(src_phase), str(src_rollout) + '.h5')
        metadata = load_data(self.data_names, data_path)
        # load images for dynamics prediction
        for i in range(self.length):
            path = os.path.join(self.dataf, str(src_phase), str(src_rollout), 'fig_%d%s' % (src_timestep + i, suffix))
            img = self.loader(path)
            img = resize_and_crop(self.phase, img, self.scale_size, self.crop_size)
            img = self.trans_to_tensor(img)
            imgs.append(img)
        imgs = torch.stack(imgs, 0)
        
        # get ground truth edge type
        edge_type = metadata[3][0, :, 0].astype(np.int)
        edge_attr = metadata[3][0, :, 1:]
        edge_type_gt = np.zeros((args.n_ball, args.n_ball, args.edge_type_num))
        edge_attr_gt = np.zeros((args.n_ball, args.n_ball, edge_attr.shape[1]))
        cnt = 0
        for x in range(args.n_ball):
            for y in range(x):
                edge_type_gt[x, y, edge_type[cnt]] = 1.
                edge_type_gt[y, x, edge_type[cnt]] = 1.
                edge_attr_gt[x, y] = edge_attr[cnt]
                edge_attr_gt[y, x] = edge_attr[cnt]
                cnt += 1

        edge_type_gt = torch.FloatTensor(edge_type_gt)
        edge_attr_gt = torch.FloatTensor(edge_attr_gt)
        graph_gt = edge_type_gt, edge_attr_gt

        # get ground truth states
        states = metadata[1] / 80.
        kps_gt = states[src_timestep:src_timestep + self.length, :, :4]
        kps_gt[:, :, 1] *= -1
        kps_gt = torch.FloatTensor(kps_gt)
        actions = metadata[2] / 600.
        actions = actions[src_timestep:src_timestep + self.length]
        actions = torch.FloatTensor(actions)
        return imgs, kps_gt, graph_gt, actions, src_phase
                  
    def __getitem__(self, idx):

        imgs, kps_gt, graph_gt, actions, src_phase = self.retrieve_by_index(idx)
        idx_rnd = random.randint(0, self.__len__() -1)
        imgsr, kps_gtr, graph_gtr, actionsr, src_phaser = self.retrieve_by_index(idx_rnd)

        sample = {"s1": {"yt": kps_gt, 
                         "xt": imgs,
                         "graph_gt": graph_gt,
                         "actions": actions,
                         "ct": int(src_phase)},
                  "s2": {"yt": kps_gtr, 
                         "xt": imgsr,
                         "graph_gt": graph_gtr,
                         "actions": actionsr,
                         "ct": int(src_phaser)}
                  }

        return sample