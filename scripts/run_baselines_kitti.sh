#/bin/bash
# Linear
python train_baselines_kitti.py \
    --exp kittimask_bvae \
    --seed 770

python train_baselines_kitti.py \
    --exp kittimask_fvae \
    --seed 770

python train_kitti_mask_svae.py \
    --exp kittimask_svae \
    --seed 770

python train_kitti_mask_pcl.py \
    --exp kittimask_pcl \
    --seed 770

python train_baselines_kitti.py \
    --exp kittimask_bvae \
    --seed 100

python train_baselines_kitti.py \
    --exp kittimask_fvae \
    --seed 100

python train_kitti_mask_svae.py \
    --exp kittimask_svae \
    --seed 100

python train_kitti_mask_pcl.py \
    --exp kittimask_pcl \
    --seed 100