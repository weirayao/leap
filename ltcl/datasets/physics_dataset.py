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
import scipy.misc
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset

from ltcl.tools.physics_engine import BallEngine, ClothEngine

from ltcl.tools.utils import rand_float, rand_int
from ltcl.tools.utils import init_stat, combine_stat, load_data, store_data
from ltcl.tools.utils import resize, crop
from ltcl.tools.utils import adjust_brightness, adjust_saturation, adjust_contrast, adjust_hue
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

        engine.init(n_ball)

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
            act += np.random.laplace(loc=0.0, scale=0.5, size=(5,2)) * 300 - act * 0.1 - state[:, 2:] * 0.1
            # act += (np.random.rand(n_obj, 2) - 0.5) * 600 - act * 0.1 - state[:, 2:] * 0.1
            act = np.clip(act, -1000, 1000)
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


def gen_Cloth(info):
    env, env_idx = info['env'], info['env_idx']
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    dt, args, phase = info['dt'], info['args'], info['phase']
    vis_width, vis_height = info['vis_width'], info['vis_height']

    state_dim = args.state_dim
    action_dim = args.action_dim
    dt = 1. / 60.

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    stats = [init_stat(state_dim), init_stat(action_dim)]

    engine = ClothEngine(dt, state_dim, action_dim)

    import pyflex
    pyflex.init()

    for i in tqdm(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        engine.init(pyflex)

        scene_params = engine.scene_params

        action = np.zeros(4)
        states_all = np.zeros((time_step, engine.n_particles, state_dim))
        actions_all = np.zeros((time_step, 1, action_dim))

        # drop the cloth down
        engine.set_action(action)
        engine.step()

        for j in range(time_step):
            positions = pyflex.get_positions().reshape(-1, 4)[:, :3]

            # sample the action
            if j % 5 == 0:
                ctrl_pts = rand_int(0, 8)

                act_lim = 0.05
                dx = rand_float(-act_lim, act_lim)
                dz = rand_float(-act_lim, act_lim)
                dy = 0.05

                action = np.array([ctrl_pts, dx, dy, dz])

            else:
                action[2] = 0.

            # store the rollout information
            state = engine.get_state()
            states_all[j] = state

            tga_path = os.path.join(rollout_dir, '%d.tga' % j)
            pyflex.render(capture=True, path=tga_path)
            tga = Image.open(tga_path)
            tga = np.array(tga)[:, 60:780, :3][:, :, ::-1]
            tga = cv2.resize(tga, (vis_width, vis_height), interpolation=cv2.INTER_AREA)
            os.system('rm ' + tga_path)

            jpg_path = os.path.join(rollout_dir, 'fig_%d.jpg' % j)
            cv2.imwrite(jpg_path, tga)

            actions_all[j, 0] = action.copy()

            engine.set_action(action)
            engine.step()

        datas = [states_all, actions_all, scene_params]
        store_data(data_names, datas, rollout_dir + '.h5')

        datas = [datas[j].astype(np.float64) for j in range(len(datas))]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    pyflex.clean()

    return stats


class PhysicsDataset(Dataset):

    def __init__(self, args, phase, trans_to_tensor=None, loader=default_loader):

        self.args = args
        self.phase = phase
        self.trans_to_tensor = trans_to_tensor
        self.loader = loader

        dataf = os.path.join(args.root_dir, args.dataf + '_' + args.env)
        self.data_dir = os.path.join(dataf, phase)
        self.stat_path = os.path.join(dataf, 'stat.h5')
        self.stat = None
        os.system('mkdir -p ' + self.data_dir)
        self.data_names = ['attrs', 'states', 'actions', 'rels']
        ratio = self.args.train_valid_ratio

        if phase in {'train'}:
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase in {'valid'}:
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        elif phase in {'raw'}:
            self.n_rollout = self.args.n_rollout
        else:
            raise AssertionError("Unknown phase")

        self.T = self.args.time_step
        self.scale_size = args.scale_size
        self.crop_size = args.crop_size

    def load_data(self):
        self.stat = load_data(self.data_names, self.stat_path)

    def gen_data(self):
        # if the data hasn't been generated, generate the data
        n_rollout, time_step, dt = self.n_rollout, self.args.time_step, self.args.dt
        assert n_rollout % self.args.num_workers == 0

        print("Generating data ... n_rollout=%d, time_step=%d" % (n_rollout, time_step))
        infos = []
        for i in range(self.args.num_workers):
            info = {'thread_idx': i,
                    'data_dir': self.data_dir,
                    'data_names': self.data_names,
                    'n_rollout': n_rollout // self.args.num_workers,
                    'time_step': time_step,
                    'dt': dt,
                    'video': False,
                    'phase': self.phase,
                    'args': self.args,
                    'vis_height': self.args.height_raw,
                    'vis_width': self.args.width_raw}

            if self.args.env in ['Ball']:
                info['env'] = 'Ball'
                info['n_ball'] = self.args.n_ball
            elif self.args.env in ['Cloth']:
                info['env'] = 'Cloth'
                info['env_idx'] = 15

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)
        env = self.args.env
        if env in ['Ball']:
            data = pool.map(gen_Ball, infos)
        elif env in ['Cloth']:
            data = pool.map(gen_Cloth, infos)
        else:
            raise AssertionError("Unknown env")

        print("Training data generated, warpping up stats ...")

        if self.phase in ('train', 'raw'):
            if env in ['Ball']:
                self.stat = [init_stat(self.args.attr_dim),
                             init_stat(self.args.state_dim),
                             init_stat(self.args.action_dim)]
            elif env in ['Cloth']:
                self.stat = [init_stat(self.args.state_dim),
                             init_stat(self.args.action_dim)]

            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])

            store_data(self.data_names[:len(self.stat)], self.stat, self.stat_path)

        else:
            print("Loading stat from %s ..." % self.stat_path)
            self.stat = load_data(self.data_names, self.stat_path)

    def __len__(self):
        args = self.args
        length = self.n_rollout * (args.time_step - 3)
        return length

    def __getitem__(self, idx):
        args = self.args
        suffix = '.png'
        offset = args.time_step - 3
        src_rollout = idx // offset
        src_timestep = idx % offset
        '''
        used for dynamics modeling
        '''
        imgs = []
        data_path = os.path.join(self.data_dir, str(src_rollout) + '.h5')
        metadata = load_data(self.data_names, data_path)
        # load images for dynamics prediction
        for i in range(4):
            path = os.path.join(self.data_dir, str(src_rollout), 'fig_%d%s' % (src_timestep + i, suffix))
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
        kps_gt = states[src_timestep:src_timestep + 4, :, :4]
        kps_gt[:, :, 1] *= -1
        kps_gt = torch.FloatTensor(kps_gt)
        actions = metadata[2] / 600.
        actions = actions[src_timestep:src_timestep + 4]
        actions = torch.FloatTensor(actions)

        return imgs, kps_gt, graph_gt, actions
