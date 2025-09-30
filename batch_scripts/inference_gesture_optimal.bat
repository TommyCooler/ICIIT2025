@echo off
REM Inference for Gesture recognition anomaly detection
REM Using optimal configuration parameters

echo ========================================
echo Running Inference for GESTURE
echo ========================================

REM Change to parent directory
cd ..

python src/inference/inference.py ^
    --dataset gesture ^
    --data_path "datasets/gesture" ^
    --window_size 80 ^
    --batch_size 48 ^
    --mask_mode time ^
    --mask_ratio 0.15 ^
    --save_excel ^
    --save_plot

echo ========================================
echo Inference completed for GESTURE
echo Check results in inference_results_gesture
echo ========================================
pause
