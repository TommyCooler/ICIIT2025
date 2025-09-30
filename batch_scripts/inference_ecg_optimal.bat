@echo off
REM Inference for ECG anomaly detection with cardiac signal data
REM Using optimal configuration parameters

echo ========================================
echo Running Inference for ECG
echo ========================================

REM Change to parent directory
cd ..

python src/inference/inference.py ^
    --dataset ecg ^
    --data_path "datasets/ecg" ^
    --dataset_name chfdb_chf01_275.pkl ^
    --window_size 64 ^
    --batch_size 64 ^
    --mask_mode time ^
    --mask_ratio 0.15 ^
    --save_excel ^
    --save_plot

echo ========================================
echo Inference completed for ECG
echo Check results in inference_results_ecg
echo ========================================
pause
