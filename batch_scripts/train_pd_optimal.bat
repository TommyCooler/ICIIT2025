@echo off
REM Process data anomaly detection
REM Optimal configuration for PD dataset
REM Conservative augmentation to preserve important information

echo ========================================
echo Training PD with Optimal Configuration
echo ========================================

REM Change to parent directory
cd ..

python src/model/main.py ^
    --dataset pd ^
    --data_path "datasets/pd" ^
    --d_model 256 ^
    --projection_dim 128 ^
    --nhead 6 ^
    --transformer_layers 5 ^
    --tcn_kernel_size 3 ^
    --tcn_num_layers 4 ^
    --dropout 0.1 ^
    --temperature 0.8 ^
    --combination_method stack ^
    --aug_nhead 3 ^
    --aug_num_layers 1 ^
    --aug_tcn_kernel_size 3 ^
    --aug_tcn_num_layers 1 ^
    --aug_dropout 0.1 ^
    --window_size 150 ^
    --batch_size 32 ^
    --num_epochs 100 ^
    --learning_rate 0.0001 ^
    --weight_decay 1e-05 ^
    --contrastive_weight 1.0 ^
    --reconstruction_weight 1.0 ^
    --epsilon 1e-05 ^
    --mask_mode time ^
    --mask_ratio 0.2 ^
    --save_dir checkpoints_pd_optimal ^
    --save_every 10 ^
    --device cuda ^
    --num_workers 4 ^
    --seed 42 ^
    --use_contrastive ^
    --use_wandb ^
    --use_lr_scheduler

echo ========================================
echo Training completed for PD
echo Check results in checkpoints_pd_optimal
echo ========================================
pause
