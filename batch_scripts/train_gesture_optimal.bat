@echo off
REM Gesture recognition anomaly detection
REM Optimal configuration for GESTURE dataset
REM Conservative augmentation to preserve important information

echo ========================================
echo Training GESTURE with Optimal Configuration
echo ========================================

REM Change to parent directory
cd ..

python src/model/main.py ^
    --dataset gesture ^
    --data_path "datasets/gesture" ^
    --d_model 256 ^
    --projection_dim 128 ^
    --nhead 8 ^
    --transformer_layers 4 ^
    --tcn_kernel_size 4 ^
    --tcn_num_layers 4 ^
    --dropout 0.1 ^
    --temperature 0.7 ^
    --combination_method stack ^
    --aug_nhead 2 ^
    --aug_num_layers 1 ^
    --aug_tcn_kernel_size 3 ^
    --aug_tcn_num_layers 1 ^
    --aug_dropout 0.1 ^
    --window_size 80 ^
    --batch_size 48 ^
    --num_epochs 100 ^
    --learning_rate 0.0002 ^
    --weight_decay 1e-05 ^
    --contrastive_weight 0.8 ^
    --reconstruction_weight 1.2 ^
    --epsilon 1e-05 ^
    --mask_mode time ^
    --mask_ratio 0.15 ^
    --save_dir checkpoints_gesture_optimal ^
    --save_every 10 ^
    --device cuda ^
    --num_workers 4 ^
    --seed 42 ^
    --use_contrastive ^
    --use_wandb ^
    --use_lr_scheduler

echo ========================================
echo Training completed for GESTURE
echo Check results in checkpoints_gesture_optimal
echo ========================================
pause
