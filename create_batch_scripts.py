#!/usr/bin/env python3
"""
Generate batch training scripts for all datasets with optimal configurations
"""

import os
from src.utils.optimal_configs import OptimalConfigs

def create_batch_scripts():
    """Create batch training scripts for all datasets"""
    
    datasets = {
        'ecg': {
            'dataset_name': 'chfdb_chf01_275.pkl',
            'description': 'ECG anomaly detection with cardiac signal data'
        },
        'ucr': {
            'dataset_name': '135',
            'description': 'UCR time series anomaly detection'
        },
        'psm': {
            'dataset_name': None,
            'description': 'PSM multivariate anomaly detection'
        },
        'gesture': {
            'dataset_name': None,
            'description': 'Gesture recognition anomaly detection'
        },
        'pd': {
            'dataset_name': None,
            'description': 'Process data anomaly detection'
        }
    }
    
    # Create batch scripts directory
    batch_dir = "batch_scripts"
    os.makedirs(batch_dir, exist_ok=True)
    
    for dataset_type, info in datasets.items():
        # Get optimal config
        config = OptimalConfigs.get_config(dataset_type)
        
        # Create training script
        script_content = f"""@echo off
REM {info['description']}
REM Optimal configuration for {dataset_type.upper()} dataset
REM Conservative augmentation to preserve important information

echo ========================================
echo Training {dataset_type.upper()} with Optimal Configuration
echo ========================================

python src/model/main.py ^
    --dataset {config['dataset']} ^
    --data_path "{config.get('data_path', f'datasets/{dataset_type}')}" ^"""

        # Add dataset name if specified
        if info['dataset_name']:
            script_content += f"""
    --dataset_name {info['dataset_name']} ^"""
        
        # Add input_dim if specified
        if config['input_dim']:
            script_content += f"""
    --input_dim {config['input_dim']} ^"""
        
        script_content += f"""
    --d_model {config['d_model']} ^
    --projection_dim {config['projection_dim']} ^
    --nhead {config['nhead']} ^
    --transformer_layers {config['transformer_layers']} ^
    --tcn_kernel_size {config['tcn_kernel_size']} ^
    --tcn_num_layers {config['tcn_num_layers']} ^
    --dropout {config['dropout']} ^
    --temperature {config['temperature']} ^
    --combination_method {config['combination_method']} ^
    --aug_nhead {config['aug_nhead']} ^
    --aug_num_layers {config['aug_num_layers']} ^
    --aug_tcn_kernel_size {config['aug_tcn_kernel_size']} ^
    --aug_tcn_num_layers {config['aug_tcn_num_layers']} ^
    --aug_dropout {config['aug_dropout']} ^
    --window_size {config['window_size']} ^
    --batch_size {config['batch_size']} ^
    --num_epochs {config['num_epochs']} ^
    --learning_rate {config['learning_rate']} ^
    --weight_decay {config['weight_decay']} ^
    --contrastive_weight {config['contrastive_weight']} ^
    --reconstruction_weight {config['reconstruction_weight']} ^
    --epsilon {config['epsilon']} ^
    --mask_mode {config['mask_mode']} ^
    --mask_ratio {config['mask_ratio']} ^
    --save_dir {config['save_dir']} ^
    --save_every {config['save_every']} ^
    --device {config['device']} ^
    --num_workers {config['num_workers']} ^
    --seed {config['seed']}"""

        # Add boolean flags
        if config.get('use_contrastive', True):
            script_content += " ^\n    --use_contrastive"
        if config.get('use_wandb', True):
            script_content += " ^\n    --use_wandb"
        if config.get('use_lr_scheduler', True):
            script_content += " ^\n    --use_lr_scheduler"

        script_content += f"""

echo ========================================
echo Training completed for {dataset_type.upper()}
echo Check results in {config['save_dir']}
echo ========================================
pause
"""
        
        # Save training script
        train_script_path = os.path.join(batch_dir, f"train_{dataset_type}_optimal.bat")
        with open(train_script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Created training script: {train_script_path}")
        
        # Create inference script
        inference_content = f"""@echo off
REM Inference for {info['description']}
REM Using optimal configuration parameters

echo ========================================
echo Running Inference for {dataset_type.upper()}
echo ========================================

python src/inference/inference.py ^
    --dataset {config['dataset']} ^
    --data_path "{config.get('data_path', f'datasets/{dataset_type}')}" ^"""

        if info['dataset_name']:
            inference_content += f"""
    --dataset_name {info['dataset_name']} ^"""
        
        inference_content += f"""
    --window_size {config['window_size']} ^
    --batch_size {config['batch_size']} ^
    --mask_mode {config['mask_mode']} ^
    --mask_ratio {config['mask_ratio']} ^
    --save_excel ^
    --save_plot

echo ========================================
echo Inference completed for {dataset_type.upper()}
echo Check results in inference_results_{dataset_type}
echo ========================================
pause
"""
        
        # Save inference script
        inference_script_path = os.path.join(batch_dir, f"inference_{dataset_type}_optimal.bat")
        with open(inference_script_path, 'w') as f:
            f.write(inference_content)
        
        print(f"Created inference script: {inference_script_path}")
    
    # Create master script to run all
    master_content = """@echo off
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
"""
    
    master_script_path = os.path.join(batch_dir, "run_all_optimal.bat")
    with open(master_script_path, 'w') as f:
        f.write(master_content)
    
    print(f"Created master script: {master_script_path}")
    
    # Create README for batch scripts
    readme_content = f"""# Optimal Training Scripts

This directory contains optimized training and inference scripts for all datasets.

## Key Features:
- **Conservative Augmentation**: Preserves important information across all datasets
- **Dataset-Specific Optimization**: Tailored parameters for each dataset's characteristics
- **Automated Execution**: Ready-to-run batch scripts

## Scripts:

### Training Scripts:
- `train_ecg_optimal.bat` - ECG dataset (2 features, cardiac signals)
- `train_ucr_optimal.bat` - UCR dataset (1 feature, sparse anomalies)
- `train_psm_optimal.bat` - PSM dataset (26 features, multivariate)
- `train_gesture_optimal.bat` - Gesture dataset (variable features)
- `train_pd_optimal.bat` - PD dataset (process data)

### Inference Scripts:
- `inference_ecg_optimal.bat` - ECG inference
- `inference_ucr_optimal.bat` - UCR inference  
- `inference_psm_optimal.bat` - PSM inference
- `inference_gesture_optimal.bat` - Gesture inference
- `inference_pd_optimal.bat` - PD inference

### Master Script:
- `run_all_optimal.bat` - Interactive menu to run any configuration

## Configuration Highlights:

| Dataset | Features | d_model | Window | Batch | Aug Strategy |
|---------|----------|---------|--------|-------|--------------|
| ECG     | 2        | 128     | 64     | 64    | Conservative |
| UCR     | 1        | 64      | 100    | 128   | Minimal      |
| PSM     | 26       | 384     | 150    | 24    | Gentle       |
| Gesture | Variable | 256     | 80     | 48    | Standard     |
| PD      | Variable | 256     | 150    | 32    | Balanced     |

## Augmentation Philosophy:
All configurations use **conservative augmentation** to preserve important information:
- Low dropout (0.05-0.1) to maintain signal integrity
- Simple architecture (1 layer) to avoid over-transformation
- Standard kernel sizes to preserve temporal patterns
- No temperature randomness in augmentation

## Usage:
1. Double-click any `.bat` file to run
2. Or use the master script `run_all_optimal.bat` for interactive selection
3. Monitor training progress and check results in respective checkpoint directories

## Requirements:
- Python environment with all dependencies installed
- CUDA-capable GPU (recommended)
- Sufficient disk space for checkpoints and results
"""
    
    readme_path = os.path.join(batch_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README: {readme_path}")
    print(f"\nAll scripts created in '{batch_dir}' directory")
    print("You can now run any script by double-clicking or use the master script for interactive selection.")

if __name__ == "__main__":
    create_batch_scripts()