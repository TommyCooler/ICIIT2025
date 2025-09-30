@echoecho ========================================
echo Training ECG with Optimal Configuration
echo ========================================

REM Change to parent directory
cd ..

python src/model/main.py ^
REM ECG anomaly detection with cardiac signal data
REM Optimal configuration for ECG dataset
REM Conservative augmentation to preserve important information

echo ========================================
echo Training ECG with Optimal Configuration
echo ========================================

python src\model\main.py ^
    --dataset ecg ^
    --data_path "datasets\ecg" ^
    --dataset_name chfdb_chf01_275.pkl ^
    --input_dim 2 ^
    --d_model 128 ^
    --projection_dim 64 ^
    --nhead 4 ^
    --transformer_layers 4 ^
    --tcn_kernel_size 3 ^
    --tcn_num_layers 3 ^
    --dropout 0.1 ^
    --temperature 0.7 ^
    --combination_method stack ^
    --aug_nhead 2 ^
    --aug_num_layers 1 ^
    --aug_tcn_kernel_size 3 ^
    --aug_tcn_num_layers 1 ^
    --aug_dropout 0.1 ^
    --window_size 64 ^
    --batch_size 64 ^
    --num_epochs 100 ^
    --learning_rate 0.0003 ^
    --weight_decay 1e-05 ^
    --contrastive_weight 0.8 ^
    --reconstruction_weight 1.2 ^
    --epsilon 1e-05 ^
    --mask_mode time ^
    --mask_ratio 0.15 ^
    --save_dir checkpoints_ecg_optimal ^
    --save_every 10 ^
    --device cuda ^
    --num_workers 4 ^
    --seed 42 ^
    --use_contrastive ^
    --use_wandb ^
    --use_lr_scheduler

echo ========================================
echo Training completed for ECG
echo Check results in checkpoints_ecg_optimal
echo ========================================
pause
