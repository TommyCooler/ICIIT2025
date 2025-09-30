# Optimal Training Scripts

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
