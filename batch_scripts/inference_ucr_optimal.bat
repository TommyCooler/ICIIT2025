@echo off
REM Inference for UCR time series anomaly detection
REM Using optimal configuration parameters

echo ========================================
echo Running Inference for UCR
echo ========================================

REM Change to parent directory
cd ..

python src/inference/inference.py ^
    --dataset ucr ^
    --data_path "datasets/ucr" ^
    --dataset_name 135 ^
    --window_size 100 ^
    --batch_size 128 ^
    --mask_mode time ^
    --mask_ratio 0.1 ^
    --save_excel ^
    --save_plot

echo ========================================
echo Inference completed for UCR
echo Check results in inference_results_ucr
echo ========================================
pause
