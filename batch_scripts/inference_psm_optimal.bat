@echo off
REM Inference for PSM multivariate anomaly detection
REM Using optimal configuration parameters

echo ========================================
echo Running Inference for PSM
echo ========================================

REM Change to parent directory
cd ..

python src/inference/inference.py ^
    --dataset psm ^
    --data_path "datasets/psm" ^
    --window_size 150 ^
    --batch_size 24 ^
    --mask_mode feature ^
    --mask_ratio 0.15 ^
    --save_excel ^
    --save_plot

echo ========================================
echo Inference completed for PSM
echo Check results in inference_results_psm
echo ========================================
pause
