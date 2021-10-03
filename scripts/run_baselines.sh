#/bin/bash
# Linear
python train_baselines.py \
    --exp toy_linear_ts_bvae \
    --seed 770

python train_baselines.py \
    --exp toy_linear_ts_fvae \
    --seed 770

python train_baselines.py \
    --exp toy_linear_ts_svae \
    --seed 770

python train_linear_pcl.py \
    --exp toy_linear_ts_pcl \
    --seed 770

# Nonlinear
python train_baselines.py \
    --exp toy_nonlinear_ts_bvae \
    --seed 770

python train_baselines.py \
    --exp toy_nonlinear_ts_fvae \
    --seed 770

python train_baselines.py \
    --exp toy_nonlinear_ts_svae \
    --seed 770

python train_baselines.py \
    --exp toy_nonlinear_ts_ivae \
    --seed 770

python train_baselines.py \
    --exp toy_nonlinear_ts_tcl \
    --seed 770

python train_nonlinear_pcl.py \
    --exp toy_nonlinear_ts_pcl \
    --seed 770


# Linear
python train_baselines.py \
    --exp toy_linear_ts_bvae \
    --seed 100

python train_baselines.py \
    --exp toy_linear_ts_fvae \
    --seed 100

python train_baselines.py \
    --exp toy_linear_ts_svae \
    --seed 100

python train_linear_pcl.py \
    --exp toy_linear_ts_pcl \
    --seed 100

# Nonlinear
python train_baselines.py \
    --exp toy_nonlinear_ts_bvae \
    --seed 100

python train_baselines.py \
    --exp toy_nonlinear_ts_fvae \
    --seed 100

python train_baselines.py \
    --exp toy_nonlinear_ts_svae \
    --seed 100

python train_baselines.py \
    --exp toy_nonlinear_ts_ivae \
    --seed 100

python train_baselines.py \
    --exp toy_nonlinear_ts_tcl \
    --seed 100

python train_nonlinear_pcl.py \
    --exp toy_nonlinear_ts_pcl \
    --seed 100


