import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectory(results_dict, benchmark_name, save_dir="experiments"):
    """
    Plots the optimization trajectory (best value found over time).
    """
    plt.figure(figsize=(10, 6))
    
    # Collect all values to decide on scaling later
    all_values = []

    for label, costs in results_dict.items():
        # Calculate trajectory: running minimum
        trajectory = []
        current_best = np.inf
        
        for c in costs:
            if c < current_best:
                current_best = c
            trajectory.append(current_best)
            all_values.append(current_best)
            
        plt.plot(trajectory, label=label, linewidth=2)

    plt.title(f"Optimization Trajectory: {benchmark_name}")
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Best Cost Found")
    
    # If any value is <= 0, we cannot use log scale. Switch to linear.
    if any(v <= 0 for v in all_values):
        plt.yscale("linear")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
    else:
        plt.yscale("log")
        plt.grid(True, which="both", ls="-", alpha=0.3)
    # -------------------------

    plt.legend()
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{save_dir}/{benchmark_name}_comparison.png"
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")