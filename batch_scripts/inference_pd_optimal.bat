@echo off
REM Inference for Process data anomaly detection
REM Using optimal configuration parameters

echo ========================================
echo Running Inference for PD
echo ========================================

REM Change to parent directory
cd ..

python src/inference/inference.py ^
    --dataset pd ^
    --data_path "datasets/pd" ^
    --window_size 150 ^
    --batch_size 32 ^
    --mask_mode time ^
    --mask_ratio 0.2 ^
    --save_excel ^
    --save_plot

echo ========================================
echo Inference completed for PD
echo Check results in inference_results_pd
echo ========================================
pause
