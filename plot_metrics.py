import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import s3fs

def download_metrics_from_s3(s3_folder_path, profile=None):
    """
    Downloads all JSON metrics files from an S3 folder and returns a list of dictionaries.
    
    Args:
        s3_folder_path (str): The S3 path (e.g., 'ubs-homes/erasmus/elouann/experiments/10')
        profile (str, optional): AWS profile name if required.
        
    Returns:
        list: List of loaded metrics dictionaries.
    """
    # Initialize s3fs
    fs = s3fs.S3FileSystem(anon=False) 
    
    # Ensure the path doesn't have the s3:// prefix if s3fs doesn't expect it
    s3_folder_path = s3_folder_path.replace('s3://', '')
    
    print(f"Searching for metrics in: {s3_folder_path}")
    
    # List all files in the directory
    all_files = fs.ls(s3_folder_path)
    metrics_files = [f for f in all_files if f.endswith('.json')]
    
    all_metrics = []
    
    for file_path in metrics_files:
        try:
            with fs.open(file_path, 'r') as f:
                data = json.load(f)
                # Keep track of the filename in the dict for debugging
                data['filename'] = os.path.basename(file_path)
                all_metrics.append(data)
                print(f"Loaded: {data['filename']}")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            
    return all_metrics

def plot_speedup_curves(directory='./metrics'):
    """Plot speedup curves from JSON metrics files in the specified directory.

    Args:
        directory (str, optional): Directory containing JSON metrics files. Defaults to './metrics'.
    """
    # 1. Load all JSON metrics files
    files = glob.glob(os.path.join(directory, "metrics_*.json"))
    all_data = []
    for f in files:
        with open(f, 'r') as file:
            all_data.append(json.load(file))

    if not all_data:
        print("No metrics files found.")
        return

    # 2. Extract unique fractions and epochs
    # We plot different lines for different fractions
    fractions = sorted(list(set(d['fraction'] for d in all_data)))
    
    plt.figure(figsize=(12, 8))

    for frac in fractions:
        # Filter data for this fraction (assuming same epoch count for comparison)
        subset = [d for d in all_data if d['fraction'] == frac]
        
        # Determine number of GPUs and training time
        plot_points = []
        for d in subset:
            # Calculate GPU count from the list in your dict
            num_gpus = len(d['gpus_used']) if isinstance(d['gpus_used'], list) else 1
            train_time = d['performance']['total_train_time_sec']
            plot_points.append((num_gpus, train_time))
        
        # Sort by GPU count (X-axis)
        plot_points.sort() 
        
        # Calculate Speedup relative to the 1-GPU baseline of THIS fraction
        # Speedup = Time(1 GPU) / Time(N GPUs)
        base_gpu_count, base_time = plot_points[0]
        
        if base_gpu_count != 1:
            print(f"Warning: No 1-GPU baseline for fraction {frac}%. Scaling might be inaccurate.")

        gpu_counts = [p[0] for p in plot_points]
        speedups = [base_time / p[1] for p in plot_points]

        # Plot actual scaling line
        plt.plot(gpu_counts, speedups, marker='o', markersize=8, label=f'Fraction: {frac}%', linewidth=2.5)

    # 3. Add the Ideal Scaling Line (y = x)
    max_gpus = max([len(d['gpus_used']) if isinstance(d['gpus_used'], list) else 1 for d in all_data])
    ideal_x = np.linspace(1, max_gpus, 10)
    plt.plot(ideal_x, ideal_x, color='red', linestyle='--', alpha=0.7, label='Ideal Scaling (Linear)')

    # 4. Final Formatting
    plt.title("BigEarthNet: Multi-GPU Speedup Curve", fontsize=16, fontweight='bold')
    plt.xlabel("Number of GPUs", fontsize=14)
    plt.ylabel("Speedup Factor (relative to 1 GPU)", fontsize=14)
    
    plt.xticks(range(1, max_gpus + 1))
    plt.yticks(range(1, max_gpus + 1))
    
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(fontsize=12)
    
    # Save the plot
    output_name = "speedup_scaling_curve.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Speedup curve saved as {output_name}")

def plot_training_accuracy_comparison(directory='./metrics'):
    # 1. Load all JSON metrics files
    files = glob.glob(os.path.join(directory, "metrics_*.json"))
    all_data = []
    for f in files:
        with open(f, 'r') as file:
            all_data.append(json.load(file))

    if not all_data:
        print("No metrics files found.")
        return

    # 2. Group experiments by Fraction
    fractions = sorted(list(set(d['fraction'] for d in all_data)))
    num_fractions = len(fractions)

    # Setup the figure grid
    fig, axes = plt.subplots(1, num_fractions, figsize=(6 * num_fractions, 5))
    if num_fractions == 1:
        axes = [axes]

    for idx, frac in enumerate(fractions):
        ax = axes[idx]
        subset = [d for d in all_data if d['fraction'] == frac]
        
        # Sort subset by GPU count for consistent legend
        subset.sort(key=lambda d: len(d['gpus_used']) if isinstance(d['gpus_used'], list) else 1)

        for data in subset:
            num_gpus = len(data['gpus_used']) if isinstance(data['gpus_used'], list) else 1
            strat = data['strategy']
            acc_history = data['training_history']['accuracy']
            
            label = f"{strat} ({num_gpus} GPUs)"
            ax.plot(range(1, len(acc_history) + 1), acc_history, marker='.', label=label)

        ax.set_title(f"Fraction: {frac}%", fontsize=14, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=12)
        if idx == 0 or idx == 2:
            ax.set_ylabel("Training Accuracy", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)

    plt.suptitle("Training Accuracy Convergence across Fractions and GPU Clusters", fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Save the result
    output_path = "accuracy_comparison_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Accuracy curves saved to {output_path}")


if __name__ == "__main__":
    plot_speedup_curves()
    plot_training_accuracy_comparison()
