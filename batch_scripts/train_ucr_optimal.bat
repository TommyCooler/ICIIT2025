@echo off
REM UCR time series anomaly detection
REM Optimal configuration for UCR dataset
REM Conservative augmentation to preserve important information

echo ========================================
echo Training UCR with Optimal Configuration
echo ========================================

REM Change to parent directory
cd ..

python src/model/main.py ^
    --dataset ucr ^
    --data_path "datasets/ucr" ^
    --dataset_name 135 ^
    --input_dim 1 ^
    --d_model 64 ^
    --projection_dim 32 ^
    --nhead 2 ^
    --transformer_layers 2 ^
    --tcn_kernel_size 5 ^
    --tcn_num_layers 2 ^
    --dropout 0.15 ^
    --temperature 0.5 ^
    --combination_method stack ^
    --aug_nhead 1 ^
    --aug_num_layers 1 ^
    --aug_tcn_kernel_size 3 ^
    --aug_tcn_num_layers 1 ^
    --aug_dropout 0.05 ^
    --window_size 100 ^
    --batch_size 128 ^
    --num_epochs 150 ^
    --learning_rate 0.0005 ^
    --weight_decay 1e-05 ^
    --contrastive_weight 0.3 ^
    --reconstruction_weight 1.7 ^
    --epsilon 1e-05 ^
    --mask_mode time ^
    --mask_ratio 0.1 ^
    --save_dir checkpoints_ucr_optimal ^
    --save_every 10 ^
    --device cuda ^
    --num_workers 4 ^
    --seed 42 ^
    --use_contrastive ^
    --use_wandb ^
    --use_lr_scheduler

echo ========================================
echo Training completed for UCR
echo Check results in checkpoints_ucr_optimal
echo ========================================
pause
