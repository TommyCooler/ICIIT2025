#!/usr/bin/env python3
"""
Automated training and inference script for all dataset types
Supports: ECG, PSM, NAB, SMAP-MSL, SMD, UCR, WADI, SWAT
"""

import os
import subprocess
import sys
import glob
import shutil
from pathlib import Path

# Base paths
BASE = r"/kaggle/working/ICIIT2025"
DATASETS_DIR = os.path.join("/kaggle", "input", "nguhcv", "datasets")
OUTPUT_BASE = os.path.join(BASE, "src", "inference", "inference_results")

# Dataset configurations
DATASET_CONFIGS = {
    'ecg': {
        'data_path': os.path.join(DATASETS_DIR, "ecg"),
        'train_dir': os.path.join(DATASETS_DIR, "ecg", "labeled", "train"),
        'test_dir': os.path.join(DATASETS_DIR, "ecg", "labeled", "test"),
        'file_pattern': "*.pkl",
        'output_dir': os.path.join(OUTPUT_BASE, "ecg")
    },
    'psm': {
        'data_path': os.path.join(DATASETS_DIR, "psm"),
        'train_dir': os.path.join(DATASETS_DIR, "psm"),
        'test_dir': os.path.join(DATASETS_DIR, "psm"),
        'file_pattern': "train.csv",  # PSM has single train/test files
        'output_dir': os.path.join(OUTPUT_BASE, "psm")
    }
    ,
    # 'nab': {
    #     'data_path': os.path.join(DATASETS_DIR, "nab"),
    #     'train_dir': os.path.join(DATASETS_DIR, "nab"),
    #     'test_dir': os.path.join(DATASETS_DIR, "nab"),
    #     'file_pattern': "*_train.npy",
    #     'output_dir': os.path.join(OUTPUT_BASE, "nab")
    # },
    'smap_msl': {
        'data_path': os.path.join(DATASETS_DIR, "smap_msl_"),
        'processed_dir': os.path.join(DATASETS_DIR, "smap_msl_", "processed"),
        'output_dir': os.path.join(OUTPUT_BASE, "smap_msl")
    },
    'smd': {
        'data_path': os.path.join(DATASETS_DIR, "smd"),
        'train_dir': os.path.join(DATASETS_DIR, "smd"),
        'test_dir': os.path.join(DATASETS_DIR, "smd"),
        'file_pattern': "*.npy",
        'output_dir': os.path.join(OUTPUT_BASE, "smd")
    },
    'ucr': {
        'data_path': os.path.join(DATASETS_DIR, "ucr", "labeled"),
        'train_dir': os.path.join(DATASETS_DIR, "ucr", "labeled"),
        'test_dir': os.path.join(DATASETS_DIR, "ucr", "labeled"),
        'file_pattern': "*_train.npy",
        'output_dir': os.path.join(OUTPUT_BASE, "ucr")
    }
    # ,
    # 'wadi': {
    #     'data_path': os.path.join(DATASETS_DIR, "wadi"),
    #     'train_dir': os.path.join(DATASETS_DIR, "wadi"),
    #     'test_dir': os.path.join(DATASETS_DIR, "wadi"),
    #     'file_pattern': "*.csv",  # WADI typically has CSV files
    #     'output_dir': os.path.join(OUTPUT_BASE, "wadi")
    # },
    # 'swat': {
    #     'data_path': os.path.join(DATASETS_DIR, "swat"),
    #     'train_dir': os.path.join(DATASETS_DIR, "swat"),
    #     'test_dir': os.path.join(DATASETS_DIR, "swat"),
    #     'file_pattern': "*.csv",  # SWAT typically has CSV files
    #     'output_dir': os.path.join(OUTPUT_BASE, "swat")
    # }
}

def run_command(cmd, description=""):
    """Run a command inheriting the console to preserve tqdm/wandb live rendering"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")

    # Ensure real-time flushing from child Python processes
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Inherit parent's stdout/stderr so tqdm/wandb detect a TTY and update in-place
    result = subprocess.run(cmd, shell=True, env=env)

    if result.returncode != 0:
        print(f"❌ ERROR in {description} (exit code {result.returncode})")
        return False
    else:
        print(f"✅ SUCCESS: {description}")
        return True

def get_dataset_files(dataset_type, config):
    """Get list of files to process for a dataset type"""
    files = []
    
    if dataset_type == 'ecg':
        # ECG: Get all .pkl files from train directory
        train_files = sorted(glob.glob(os.path.join(config['train_dir'], config['file_pattern'])))
        for train_file in train_files:
            file_name = os.path.basename(train_file)
            test_file = os.path.join(config['test_dir'], file_name)
            if os.path.exists(test_file):
                files.append({
                    'name': file_name,
                    'train_path': train_file,
                    'test_path': test_file
                })
    
    elif dataset_type == 'psm':
        # PSM: Single train/test file pair
        train_file = os.path.join(config['train_dir'], 'train.csv')
        test_file = os.path.join(config['test_dir'], 'test.csv')
        if os.path.exists(train_file) and os.path.exists(test_file):
            files.append({
                'name': 'psm_dataset',
                'train_path': train_file,
                'test_path': test_file
            })
    
    elif dataset_type == 'smap_msl':
        # SMAP-MSL processed mode: iterate all *_train.npy triplets in processed dir
        proc_dir = config.get('processed_dir', '')
        train_files = sorted(glob.glob(os.path.join(proc_dir, "*_train.npy")))
        for train_file in train_files:
            file_name = os.path.basename(train_file).replace('_train.npy', '')
            test_file = os.path.join(proc_dir, f"{file_name}_test.npy")
            labels_file = os.path.join(proc_dir, f"{file_name}_labels.npy")
            if os.path.exists(test_file) and os.path.exists(labels_file):
                files.append({
                    'name': file_name,
                    'train_path': train_file,
                    'test_path': test_file
                })
    
    elif dataset_type == 'smd':
        # SMD: Require train/test/labels triplets in a single directory
        base_dir = config['train_dir']
        train_files = sorted(glob.glob(os.path.join(base_dir, "*_train.npy")))
        for train_file in train_files:
            file_name = os.path.basename(train_file).replace('_train.npy', '')
            test_file = os.path.join(base_dir, f"{file_name}_test.npy")
            labels_file = os.path.join(base_dir, f"{file_name}_labels.npy")
            if os.path.exists(test_file) and os.path.exists(labels_file):
                files.append({
                    'name': file_name,
                    'train_path': train_file,
                    'test_path': test_file
                })
    
    elif dataset_type == 'ucr':
        # UCR: Require train/test/labels triplets in labeled dir
        train_files = sorted(glob.glob(os.path.join(config['train_dir'], config['file_pattern'])))
        for train_file in train_files:
            file_name = os.path.basename(train_file).replace('_train.npy', '')
            test_file = os.path.join(config['test_dir'], f"{file_name}_test.npy")
            labels_file = os.path.join(config['test_dir'], f"{file_name}_labels.npy")
            if os.path.exists(test_file) and os.path.exists(labels_file):
                files.append({
                    'name': file_name,
                    'train_path': train_file,
                    'test_path': test_file
                })
    
    elif dataset_type in ['wadi', 'swat']:
        # WADI/SWAT: Check for CSV files
        csv_files = sorted(glob.glob(os.path.join(config['train_dir'], config['file_pattern'])))
        for csv_file in csv_files:
            file_name = os.path.basename(csv_file).replace('.csv', '')
            files.append({
                'name': file_name,
                'train_path': csv_file,
                'test_path': csv_file  # Same file for train/test
            })
    
    return files

def process_dataset(dataset_type, config):
    """Process all files in a dataset type"""
    print(f"\n{'#'*80}")
    print(f"PROCESSING DATASET: {dataset_type.upper()}")
    print(f"Data path: {config['data_path']}")
    print(f"Output dir: {config['output_dir']}")
    print(f"{'#'*80}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Get files to process
    files = get_dataset_files(dataset_type, config)
    
    if not files:
        print(f"❌ No files found for dataset {dataset_type}")
        return False
    
    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f"  - {f['name']}")
    
    success_count = 0
    
    for i, file_info in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE {i}/{len(files)}: {file_info['name']}")
        print(f"{'='*60}")
        
        # 1. Train model
        train_cmd = (
            f"python {os.path.join(BASE, 'src', 'model', 'main.py')} "
            f"--dataset {dataset_type} "
            f"--data_path {config['data_path']} "
            f"--dataset_name {file_info['name']} "
        )
        
        if not run_command(train_cmd, f"Training {file_info['name']}"):
            print(f"❌ Training failed for {file_info['name']}, skipping...")
            continue
        
        # 2. Run inference
        infer_cmd = (
            f"python {os.path.join(BASE, 'src', 'inference', 'inference.py')} "
            f"--dataset {dataset_type} "
            f"--data_path {config['data_path']} "
            f"--save_excel"
        )
        
        if not run_command(infer_cmd, f"Inference {file_info['name']}"):
            print(f"❌ Inference failed for {file_info['name']}")
            continue
        
        # 3. No cleanup needed - plotting is disabled in inference.py
        print("ℹ️  Plotting disabled in inference.py - no cleanup needed")
        
        success_count += 1
        print(f"✅ Successfully processed {file_info['name']}")
    
    print(f"\n{'='*60}")
    print(f"DATASET {dataset_type.upper()} COMPLETED")
    print(f"Successfully processed: {success_count}/{len(files)} files")
    print(f"{'='*60}")
    
    return success_count > 0

def main():
    """Main function to process all datasets"""
    print("🚀 Starting automated training and inference for all datasets")
    print(f"Base directory: {BASE}")
    
    # Check if base directories exist
    if not os.path.exists(BASE):
        print(f"❌ Base directory not found: {BASE}")
        sys.exit(1)
    
    if not os.path.exists(DATASETS_DIR):
        print(f"❌ Datasets directory not found: {DATASETS_DIR}")
        sys.exit(1)
    
    # Process each dataset type
    total_success = 0
    total_datasets = len(DATASET_CONFIGS)
    
    for dataset_type, config in DATASET_CONFIGS.items():
        try:
            if process_dataset(dataset_type, config):
                total_success += 1
        except Exception as e:
            print(f"❌ Error processing dataset {dataset_type}: {e}")
            continue
    
    # No final cleanup needed - plotting is disabled in inference.py
    print(f"\n{'='*60}")
    print("PLOTTING DISABLED: No image cleanup needed")
    print(f"{'='*60}")
    
    print(f"\n{'#'*80}")
    print(f"FINAL SUMMARY")
    print(f"Successfully processed: {total_success}/{total_datasets} dataset types")
    print(f"{'#'*80}")
    
    if total_success == total_datasets:
        print("🎉 All datasets processed successfully!")
    else:
        print(f"⚠️  {total_datasets - total_success} dataset types failed to process")

if __name__ == '__main__':
    main()
