#/bin/bash
# Linear
python train_baselines_ball.py \
    --exp mbi_2lag_bvae \
    --seed 770

python train_baselines_ball.py \
    --exp mbi_2lag_fvae \
    --seed 770

python train_ball_svae.py \
    --exp mbi_2lag_svae \
    --seed 770

python train_ball_pcl.py \
    --exp mbi_2lag_pcl \
    --seed 770

python train_baselines_ball.py \
    --exp mbi_2lag_bvae \
    --seed 100

python train_baselines_ball.py \
    --exp mbi_2lag_fvae \
    --seed 100

python train_ball_svae.py \
    --exp mbi_2lag_svae \
    --seed 100

python train_ball_pcl.py \
    --exp mbi_2lag_pcl \
    --seed 100