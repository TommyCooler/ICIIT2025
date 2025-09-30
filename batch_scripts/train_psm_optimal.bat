@echo off
REM PSM multivariate anomaly detection
REM Optimal configuration for PSM dataset
REM Conservative augmentation to preserve important information

echo ========================================
echo Training PSM with Optimal Configuration
echo ========================================

REM Change to parent directory
cd ..

python src/model/main.py ^
    --dataset psm ^
    --data_path "datasets/psm" ^
    --input_dim 26 ^
    --d_model 384 ^
    --projection_dim 192 ^
    --nhead 8 ^
    --transformer_layers 6 ^
    --tcn_kernel_size 3 ^
    --tcn_num_layers 5 ^
    --dropout 0.1 ^
    --temperature 0.8 ^
    --combination_method stack ^
    --aug_nhead 2 ^
    --aug_num_layers 1 ^
    --aug_tcn_kernel_size 3 ^
    --aug_tcn_num_layers 1 ^
    --aug_dropout 0.1 ^
    --window_size 150 ^
    --batch_size 24 ^
    --num_epochs 100 ^
    --learning_rate 8e-05 ^
    --weight_decay 1e-05 ^
    --contrastive_weight 1.0 ^
    --reconstruction_weight 1.0 ^
    --epsilon 1e-05 ^
    --mask_mode feature ^
    --mask_ratio 0.15 ^
    --save_dir checkpoints_psm_optimal ^
    --save_every 10 ^
    --device cuda ^
    --num_workers 4 ^
    --seed 42 ^
    --use_contrastive ^
    --use_wandb ^
    --use_lr_scheduler

echo ========================================
echo Training completed for PSM
echo Check results in checkpoints_psm_optimal
echo ========================================
pause
