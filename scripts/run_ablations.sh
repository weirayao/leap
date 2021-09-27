#/bin/bash
# Linear
python train_ablations.py \
    --exp toy_linear_ts_ri \
    --seed 770

python train_ablations.py \
    --exp toy_linear_ts_trans \
    --seed 770

python train_ablations.py \
    --exp toy_linear_ts_flow \
    --seed 770

# Nonlinear
python train_ablations.py \
    --exp toy_nonlinear_ts_ri \
    --seed 770

python train_ablations.py \
    --exp toy_nonlinear_ts_trans \
    --seed 770

python train_ablations.py \
    --exp toy_nonlinear_ts_flow \
    --seed 770

python train_nonlinear_ns.py \
    --exp toy_nonlinear_ts_dense \
    --seed 770

# Linear
python train_ablationss.py \
    --exp toy_linear_ts_ri \
    --seed 42

python train_ablations.py \
    --exp toy_linear_ts_trans \
    --seed 42

python train_ablations.py \
    --exp toy_linear_ts_flow \
    --seed 42

# Nonlinear
python train_ablationss.py \
    --exp toy_nonlinear_ts_ri \
    --seed 42

python train_ablations.py \
    --exp toy_nonlinear_ts_trans \
    --seed 42

python train_ablations.py \
    --exp toy_nonlinear_ts_flow \
    --seed 42

python train_nonlinear_ns.py \
    --exp toy_nonlinear_ts_dense \
    --seed 42



