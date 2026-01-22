# AutoML Lab Final Project: GPyTorch MLL Benchmarking in SMAC

## Project Overview
This project integrates GPyTorch surrogate models into the SMAC3 optimization loop. It benchmarks different Marginal Log Likelihood (MLL) objectives and optimizers (Adam vs LBFGS) against standard SMAC baselines.

## Setup
1. Create environment: `conda create -n automl_proj python=3.10`
2. Activate: `conda activate automl_proj`
3. Install: `pip install -r requirements.txt`

## Running Experiments
Run the main benchmark script:
```bash
python run_benchmark.py