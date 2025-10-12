#!/usr/bin/env python3
"""
Script to plot train and test data for UCR dataset 135
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path for loading data
sys.path.append('src')

def load_ucr_135_data():
    """Load UCR dataset 135"""
    data_dir = "datasets/ucr/labeled"
    
    # Load train, test, and labels
    train_data = np.load(os.path.join(data_dir, "135_train.npy"))
    test_data = np.load(os.path.join(data_dir, "135_test.npy"))
    labels = np.load(os.path.join(data_dir, "135_labels.npy"))
    
    # Load metadata from text file
    metadata_file = os.path.join(data_dir, "135_UCR_Anomaly_InternalBleeding16_1200_4187_4199.txt")
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    return train_data, test_data, labels, metadata

def plot_ucr_135_with_matplotlib():
    """Plot UCR dataset 135 with matplotlib"""
    
    print("Loading UCR dataset 135...")
    train_data, test_data, labels, metadata = load_ucr_135_data()
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Flatten data for easier plotting
    train_values = train_data.flatten()
    test_values = test_data.flatten()
    labels_flat = labels.flatten()
    
    # Create time axes
    train_time = np.arange(len(train_values))
    test_time = np.arange(len(test_values))
    
    # Find anomaly regions
    anomaly_indices = np.where(labels_flat == 1)[0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('UCR Dataset 135 - Internal Bleeding Anomaly Detection', fontsize=16, fontweight='bold')
    
    # Plot 1: Train data
    axes[0].plot(train_time, train_values, 'b-', linewidth=1, alpha=0.8, label='Train Data')
    axes[0].set_title('Train Data (Normal)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add train statistics
    train_mean = np.mean(train_values)
    train_std = np.std(train_values)
    axes[0].axhline(y=train_mean, color='red', linestyle='--', alpha=0.7, label=f'Mean: {train_mean:.3f}')
    axes[0].axhline(y=train_mean + train_std, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {train_mean + train_std:.3f}')
    axes[0].axhline(y=train_mean - train_std, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {train_mean - train_std:.3f}')
    axes[0].legend()
    
    # Plot 2: Test data with anomaly regions highlighted
    axes[1].plot(test_time, test_values, 'g-', linewidth=1, alpha=0.8, label='Test Data')
    
    # Highlight anomaly regions
    if len(anomaly_indices) > 0:
        # Find continuous anomaly regions
        anomaly_regions = []
        start_idx = None
        for i, idx in enumerate(anomaly_indices):
            if start_idx is None:
                start_idx = idx
            elif i > 0 and idx != anomaly_indices[i-1] + 1:
                anomaly_regions.append((start_idx, anomaly_indices[i-1]))
                start_idx = idx
        if start_idx is not None:
            anomaly_regions.append((start_idx, anomaly_indices[-1]))
        
        # Highlight each anomaly region
        for start, end in anomaly_regions:
            axes[1].axvspan(start, end, alpha=0.3, color='red', label='Anomaly Region' if start == anomaly_regions[0][0] else "")
    
    axes[1].set_title('Test Data with Anomaly Regions', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Add test statistics
    test_mean = np.mean(test_values)
    test_std = np.std(test_values)
    axes[1].axhline(y=test_mean, color='blue', linestyle='--', alpha=0.7, label=f'Test Mean: {test_mean:.3f}')
    axes[1].axhline(y=train_mean, color='red', linestyle='--', alpha=0.7, label=f'Train Mean: {train_mean:.3f}')
    axes[1].legend()
    
    # Plot 3: Combined view with train and test
    axes[2].plot(train_time, train_values, 'b-', linewidth=1, alpha=0.6, label='Train Data')
    axes[2].plot(test_time, test_values, 'g-', linewidth=1, alpha=0.8, label='Test Data')
    
    # Highlight anomaly regions in combined plot
    if len(anomaly_indices) > 0:
        for start, end in anomaly_regions:
            axes[2].axvspan(start, end, alpha=0.3, color='red')
    
    axes[2].set_title('Combined View: Train + Test Data', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Add statistics text box
    stats_text = f"""Dataset Statistics:
Train: {len(train_values)} points, mean={train_mean:.3f}, std={train_std:.3f}
Test: {len(test_values)} points, mean={test_mean:.3f}, std={test_std:.3f}
Anomalies: {len(anomaly_indices)} points ({len(anomaly_indices)/len(test_values)*100:.1f}%)"""
    
    if len(anomaly_indices) > 0:
        stats_text += f"\nAnomaly Regions: {len(anomaly_regions)}"
        if len(anomaly_regions) > 0:
            stats_text += f"\nRegion 1: timesteps {anomaly_regions[0][0]}-{anomaly_regions[0][1]}"
    
    axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'ucr_135_matplotlib_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Matplotlib plot saved as: {output_file}")
    
    # Show plot
    plt.show()
    
    return train_data, test_data, labels, metadata

def create_detailed_anomaly_plot(train_data, test_data, labels):
    """Create detailed plot focusing on anomaly region"""
    
    print("\nCreating detailed anomaly region plot...")
    
    # Flatten data
    train_values = train_data.flatten()
    test_values = test_data.flatten()
    labels_flat = labels.flatten()
    
    # Find anomaly region
    anomaly_indices = np.where(labels_flat == 1)[0]
    
    if len(anomaly_indices) > 0:
        # Create detailed plot around anomaly
        start_region = max(0, anomaly_indices[0] - 50)
        end_region = min(len(test_values), anomaly_indices[-1] + 50)
        
        # Extract region data
        region_time = np.arange(start_region, end_region)
        region_values = test_values[start_region:end_region]
        region_labels = labels_flat[start_region:end_region]
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'UCR Dataset 135 - Detailed Anomaly Region Analysis\nTimesteps {start_region}-{end_region}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Full context with anomaly region highlighted
        axes[0].plot(np.arange(len(test_values)), test_values, 'g-', linewidth=0.8, alpha=0.7, label='Full Test Data')
        axes[0].axvspan(start_region, end_region, alpha=0.2, color='yellow', label='Detailed Region')
        
        # Highlight actual anomaly
        for idx in anomaly_indices:
            axes[0].axvline(x=idx, color='red', linestyle='-', alpha=0.8, linewidth=2)
        
        axes[0].set_title('Full Test Data with Anomaly Region Highlighted', fontsize=14)
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Detailed view of anomaly region
        axes[1].plot(region_time, region_values, 'g-', linewidth=2, alpha=0.8, label='Test Data')
        
        # Highlight anomaly points
        anomaly_in_region = []
        for idx in anomaly_indices:
            if start_region <= idx < end_region:
                relative_idx = idx - start_region
                axes[1].scatter(idx, region_values[relative_idx], color='red', s=50, zorder=5, label='Anomaly' if len(anomaly_in_region) == 0 else "")
                anomaly_in_region.append(relative_idx)
        
        # Add train statistics for comparison
        train_mean = np.mean(train_values)
        train_std = np.std(train_values)
        axes[1].axhline(y=train_mean, color='blue', linestyle='--', alpha=0.7, label=f'Train Mean: {train_mean:.3f}')
        axes[1].axhline(y=train_mean + 2*train_std, color='orange', linestyle='--', alpha=0.7, label=f'Train ±2σ')
        axes[1].axhline(y=train_mean - 2*train_std, color='orange', linestyle='--', alpha=0.7)
        
        axes[1].set_title('Detailed View of Anomaly Region', fontsize=14)
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Add anomaly statistics
        if len(anomaly_in_region) > 0:
            anomaly_values_in_region = region_values[anomaly_in_region]
            anomaly_stats = f"""Anomaly Statistics in Region:
Points: {len(anomaly_in_region)}
Mean: {np.mean(anomaly_values_in_region):.3f}
Std: {np.std(anomaly_values_in_region):.3f}
Min: {np.min(anomaly_values_in_region):.3f}
Max: {np.max(anomaly_values_in_region):.3f}"""
            
            axes[1].text(0.02, 0.98, anomaly_stats, transform=axes[1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                        fontsize=10)
        
        plt.tight_layout()
        
        # Save detailed plot
        detailed_output = 'ucr_135_detailed_anomaly.png'
        plt.savefig(detailed_output, dpi=300, bbox_inches='tight')
        print(f"Detailed anomaly plot saved as: {detailed_output}")
        
        plt.show()
    
    return train_data, test_data, labels

def analyze_anomaly_patterns(train_data, test_data, labels):
    """Analyze patterns in the anomaly data"""
    
    print("\n" + "="*60)
    print("ANOMALY ANALYSIS")
    print("="*60)
    
    # Basic statistics
    anomaly_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]
    
    print(f"Total test points: {len(test_data)}")
    print(f"Normal points: {len(normal_indices)} ({len(normal_indices)/len(test_data)*100:.1f}%)")
    print(f"Anomaly points: {len(anomaly_indices)} ({len(anomaly_indices)/len(test_data)*100:.1f}%)")
    
    if len(anomaly_indices) > 0:
        # Anomaly region analysis
        anomaly_regions = []
        start_idx = None
        for i, idx in enumerate(anomaly_indices):
            if start_idx is None:
                start_idx = idx
            elif i > 0 and idx != anomaly_indices[i-1] + 1:
                anomaly_regions.append((start_idx, anomaly_indices[i-1]))
                start_idx = idx
        if start_idx is not None:
            anomaly_regions.append((start_idx, anomaly_indices[-1]))
        
        print(f"\nAnomaly regions: {len(anomaly_regions)}")
        for i, (start, end) in enumerate(anomaly_regions):
            duration = end - start + 1
            print(f"  Region {i+1}: timesteps {start}-{end} (duration: {duration})")
        
        # Statistical analysis
        train_mean = np.mean(train_data)
        train_std = np.std(train_data)
        
        anomaly_values = test_data[anomaly_indices]
        normal_values = test_data[normal_indices]
        
        print(f"\nStatistical Analysis:")
        print(f"Train data: mean={train_mean:.3f}, std={train_std:.3f}")
        print(f"Normal test: mean={np.mean(normal_values):.3f}, std={np.std(normal_values):.3f}")
        print(f"Anomaly test: mean={np.mean(anomaly_values):.3f}, std={np.std(anomaly_values):.3f}")
        
        # Check if anomalies are significantly different
        normal_deviation = np.abs(normal_values - train_mean) / train_std
        anomaly_deviation = np.abs(anomaly_values - train_mean) / train_std
        
        print(f"\nDeviation from train mean (in std units):")
        print(f"Normal points: mean={np.mean(normal_deviation):.3f}, max={np.max(normal_deviation):.3f}")
        print(f"Anomaly points: mean={np.mean(anomaly_deviation):.3f}, max={np.max(anomaly_deviation):.3f}")

def create_ascii_plot(train_data, test_data, labels, width=80, height=20):
    """Create ASCII art plot of the data"""
    
    print("\n" + "="*60)
    print("ASCII VISUALIZATION")
    print("="*60)
    
    # Normalize data to fit in ASCII plot
    all_data = np.concatenate([train_data.flatten(), test_data.flatten()])
    min_val = np.min(all_data)
    max_val = np.max(all_data)
    
    # Scale to height
    train_scaled = ((train_data.flatten() - min_val) / (max_val - min_val) * (height - 1)).astype(int)
    test_scaled = ((test_data.flatten() - min_val) / (max_val - min_val) * (height - 1)).astype(int)
    
    # Create ASCII plot for train data
    print("Train Data (Normal):")
    print("Value Range: {:.3f} to {:.3f}".format(min_val, max_val))
    
    # Plot train data (first 80 points)
    train_sample = train_scaled[:min(width, len(train_scaled))]
    for y in range(height-1, -1, -1):
        line = ""
        for x, val in enumerate(train_sample):
            if val == y:
                line += "●"  # Data point
            else:
                line += " "
        if line.strip():  # Only print non-empty lines
            print(f"{y:2d}| {line}")
    
    print("   " + "-" * width)
    print("   " + "".join([str(i%10) for i in range(width)]))
    
    # Create ASCII plot for test data (sample)
    print(f"\nTest Data (First {width} points):")
    test_sample = test_scaled[:width]
    test_labels_sample = labels[:width].flatten()
    
    for y in range(height-1, -1, -1):
        line = ""
        for x, (val, label) in enumerate(zip(test_sample, test_labels_sample)):
            if val == y:
                if label == 1:  # Anomaly
                    line += "X"  # Anomaly point
                else:
                    line += "●"  # Normal point
            else:
                line += " "
        if line.strip():  # Only print non-empty lines
            print(f"{y:2d}| {line}")
    
    print("   " + "-" * width)
    print("   " + "".join([str(i%10) for i in range(width)]))
    print("Legend: ● = Normal, X = Anomaly")
    
    # Show anomaly region
    anomaly_indices = np.where(labels.flatten() == 1)[0]
    if len(anomaly_indices) > 0:
        print(f"\nAnomaly Region Details:")
        print(f"Anomaly occurs at timesteps: {anomaly_indices}")
        
        # Show values around anomaly
        if len(anomaly_indices) > 0:
            start_idx = max(0, anomaly_indices[0] - 5)
            end_idx = min(len(test_data), anomaly_indices[-1] + 6)
            print(f"\nValues around anomaly region ({start_idx}-{end_idx}):")
            for i in range(start_idx, end_idx):
                marker = " <-- ANOMALY" if i in anomaly_indices else ""
                print(f"  Timestep {i:4d}: {test_data[i][0]:.4f}{marker}")

def save_data_to_files(train_data, test_data, labels):
    """Save data to text files for external plotting"""
    
    print("\n" + "="*60)
    print("SAVING DATA TO FILES")
    print("="*60)
    
    # Save train data
    with open('ucr_135_train.txt', 'w') as f:
        f.write("# UCR Dataset 135 - Train Data\n")
        f.write("# Timestep,Value\n")
        for i, val in enumerate(train_data):
            f.write(f"{i},{val[0]:.6f}\n")
    print("Train data saved to: ucr_135_train.txt")
    
    # Save test data with labels
    with open('ucr_135_test.txt', 'w') as f:
        f.write("# UCR Dataset 135 - Test Data with Labels\n")
        f.write("# Timestep,Value,Label\n")
        for i, (val, label) in enumerate(zip(test_data, labels)):
            f.write(f"{i},{val[0]:.6f},{label[0]}\n")
    print("Test data saved to: ucr_135_test.txt")
    
    # Save summary statistics
    with open('ucr_135_summary.txt', 'w') as f:
        f.write("UCR Dataset 135 - Summary Statistics\n")
        f.write("="*50 + "\n")
        f.write(f"Train Data:\n")
        f.write(f"  Length: {len(train_data)} points\n")
        f.write(f"  Mean: {np.mean(train_data):.4f}\n")
        f.write(f"  Std: {np.std(train_data):.4f}\n")
        f.write(f"  Min: {np.min(train_data):.4f}\n")
        f.write(f"  Max: {np.max(train_data):.4f}\n\n")
        
        f.write(f"Test Data:\n")
        f.write(f"  Length: {len(test_data)} points\n")
        f.write(f"  Mean: {np.mean(test_data):.4f}\n")
        f.write(f"  Std: {np.std(test_data):.4f}\n")
        f.write(f"  Min: {np.min(test_data):.4f}\n")
        f.write(f"  Max: {np.max(test_data):.4f}\n\n")
        
        anomaly_indices = np.where(labels.flatten() == 1)[0]
        f.write(f"Anomaly Information:\n")
        f.write(f"  Anomaly points: {len(anomaly_indices)} ({len(anomaly_indices)/len(test_data)*100:.1f}%)\n")
        f.write(f"  Anomaly timesteps: {list(anomaly_indices)}\n")
        
    print("Summary saved to: ucr_135_summary.txt")

if __name__ == "__main__":
    try:
        # Plot with matplotlib
        train_data, test_data, labels, metadata = plot_ucr_135_with_matplotlib()
        
        # Create detailed anomaly plot
        create_detailed_anomaly_plot(train_data, test_data, labels)
        
        # Analyze patterns
        analyze_anomaly_patterns(train_data, test_data, labels)
        
        # Create ASCII visualization
        create_ascii_plot(train_data, test_data, labels)
        
        # Save data to files for external plotting
        save_data_to_files(train_data, test_data, labels)
        
        print("\n" + "="*60)
        print("MATPLOTLIB VISUALIZATION COMPLETED")
        print("="*60)
        print("Files created:")
        print("- ucr_135_matplotlib_plot.png: Main matplotlib plot")
        print("- ucr_135_detailed_anomaly.png: Detailed anomaly region plot")
        print("- ucr_135_train.txt: Train data for plotting")
        print("- ucr_135_test.txt: Test data with labels for plotting") 
        print("- ucr_135_summary.txt: Summary statistics")
        print("\nMatplotlib plots have been displayed and saved!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
