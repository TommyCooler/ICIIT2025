@echo off
REM Master script to run all optimal configurations

echo ========================================
echo Running ALL Datasets with Optimal Configurations
echo ========================================

echo Choose dataset to train:
echo 1. ECG (2 features, cardiac signals)
echo 2. UCR (1 feature, sparse anomalies)  
echo 3. PSM (26 features, multivariate)
echo 4. Gesture (variable features)
echo 5. PD (process data)
echo 6. All datasets (sequential)
echo 7. Compare configurations only

set /p choice="Enter choice (1-7): "

if "%choice%"=="1" call train_ecg_optimal.bat
if "%choice%"=="2" call train_ucr_optimal.bat
if "%choice%"=="3" call train_psm_optimal.bat
if "%choice%"=="4" call train_gesture_optimal.bat
if "%choice%"=="5" call train_pd_optimal.bat
if "%choice%"=="6" (
    echo Running all datasets sequentially...
    call train_ecg_optimal.bat
    call train_ucr_optimal.bat
    call train_psm_optimal.bat
    call train_gesture_optimal.bat
    call train_pd_optimal.bat
)
if "%choice%"=="7" python optimal_configs.py

echo ========================================
echo Master script completed
echo ========================================
pause
