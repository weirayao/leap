#/bin/bash
python train_baselines_mocap.py \
    --exp mocap_bvae \
    --seed 770

python train_baselines_mocap.py \
    --exp mocap_fvae \
    --seed 770

python train_baselines_mocap.py \
    --exp mocap_ivae \
    --seed 770

python train_mocap_pcl.py \
    --exp mocap_pcl \
    --seed 770

python train_baselines_mocap.py \
    --exp mocap_svae \
    --seed 770

python train_baselines_mocap.py \
    --exp mocap_tcl \
    --seed 770



python train_baselines_mocap.py \
    --exp mocap_bvae \
    --seed 100

python train_baselines_mocap.py \
    --exp mocap_fvae \
    --seed 100

python train_baselines_mocap.py \
    --exp mocap_ivae \
    --seed 100

python train_mocap_pcl.py \
    --exp mocap_pcl \
    --seed 100

python train_baselines_mocap.py \
    --exp mocap_svae \
    --seed 100

python train_baselines_mocap.py \
    --exp mocap_tcl \
    --seed 100