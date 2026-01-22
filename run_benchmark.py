import os
import shutil
import warnings
import numpy as np

# SMAC imports
from smac import Scenario
from smac.facade import BlackBoxFacade, HyperparameterOptimizationFacade

# Internal imports
from utils.benchmarks import Ackley, Rosenbrock, Hartmann6, Levy
from utils.plotting import plot_trajectory
from src.gpytorch_wrapper import GPyTorchSurrogate

# Suppress GPyTorch warnings
warnings.filterwarnings("ignore", module="gpytorch")

RESULTS_DIR = "experiments"

def run_experiment(function_cls, mll_type, optimizer_type, n_trials=30):
    """
    Runs a single SMAC optimization loop.
    """
    print(f"\n[Experiment] {function_cls.__name__} | MLL: {mll_type} | Opt: {optimizer_type}")
    
    # 1. Instantiate Benchmark
    benchmark = function_cls()
    
    # 2. Define Scenario
    scenario = Scenario(
        benchmark.get_config_space(),
        deterministic=True,
        n_trials=n_trials,
        output_directory=f"{RESULTS_DIR}/{function_cls.__name__}/{mll_type}_{optimizer_type}"
    )

    # 3. Configure SMAC
    if mll_type == "Baseline":
        smac = BlackBoxFacade(
            scenario,
            benchmark.__call__,
            overwrite=True
        )
    else:
        # Custom Facade
        class GPyTorchFacade(HyperparameterOptimizationFacade):
            @staticmethod
            def get_model(scenario, **kwargs):
                return GPyTorchSurrogate(
                    config_space=scenario.configspace,
                    mll_type=mll_type,
                    optimizer_type=optimizer_type,
                    n_epochs=50,
                    learning_rate=0.1
                )

        smac = GPyTorchFacade(
            scenario,
            benchmark.__call__,
            overwrite=True
        )

    # 4. Run Optimization
    smac.optimize()
    
    # 5. Extract Costs (The Robust "Manual" Way)
    # Instead of relying on smac.intensifier.trajectory attributes (which change often),
    # we manually calculate the "best found so far" from the raw run history.
    
    raw_costs = []
    # Get all configs that were evaluated
    for config in smac.runhistory.get_configs():
        cost = smac.runhistory.get_cost(config)
        # Ensure we don't get None values
        if cost is not None:
            raw_costs.append(cost)
    
    # Calculate the trajectory (running minimum)
    # This creates the "Best Value Found vs Time" curve
    trajectory_costs = []
    current_best = np.inf
    
    for c in raw_costs:
        if c < current_best:
            current_best = c
        trajectory_costs.append(current_best)
    
    return trajectory_costs

def main():
    if os.path.exists(RESULTS_DIR):
        try:
            shutil.rmtree(RESULTS_DIR)
        except PermissionError:
            print("Could not delete experiments folder. Please close any open images.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    benchmarks = [Ackley, Rosenbrock, Hartmann6, Levy]
    
    configurations = [
        ("ExactMLL", "Adam"),
        ("ExactMLL", "LBFGS"),
        ("LOO", "Adam"),
        ("ELBO", "Adam"),
        ("Baseline", "None")
    ]

    for bench_cls in benchmarks:
        results = {}
        
        for mll, opt in configurations:
            try:
                # trials set to 30 for speed
                costs = run_experiment(bench_cls, mll, opt, n_trials=30)
                results[f"{mll}+{opt}"] = costs
            except Exception as e:
                print(f"!!! Error running {bench_cls.__name__} {mll}+{opt}: {e}")
                import traceback
                traceback.print_exc()

        print(f"Generating plot for {bench_cls.__name__}...")
        # Since we calculated the trajectory manually in run_experiment,
        # we can pass it directly to the plotter without recalculating.
        # But our plotter expects raw costs usually. 
        # Let's verify utils/plotting.py handles this. 
        # Yes, plotting.py calculates running min again, which is harmless.
        plot_trajectory(results, bench_cls.__name__, save_dir=RESULTS_DIR)

    print("\nAll experiments finished. Check the 'experiments' folder for plots.")

if __name__ == "__main__":
    main()