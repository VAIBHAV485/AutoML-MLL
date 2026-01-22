import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectory(results_dict, benchmark_name, save_dir="experiments"):
    """
    Plots the optimization trajectory (best value found over time).
    Args:results_dict (dict): Keys are labels (e.g. "ExactMLL+Adam"), 
        Values are lists of cost values from the run.
        benchmark_name (str): Name of the function (e.g. "Ackley").
        save_dir (str): Folder to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    for label, costs in results_dict.items():
        # Convert raw costs to a best so far trajectory
        trajectory = []
        current_best = np.inf
        
        for c in costs:
            if c < current_best:
                current_best = c
            trajectory.append(current_best)
            
        plt.plot(trajectory, label=label, linewidth=2)

    plt.title(f"Optimization Trajectory on {benchmark_name}")
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Best Cost (Log Scale)")
    plt.yscale("log") # Log scale helps see differences near zero
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{save_dir}/{benchmark_name}_comparison.png"
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")