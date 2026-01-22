import torch
from torch.optim import Adam, SGD, AdamW
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.optim.fit import fit_gpytorch_mll_torch

def train_gp(model, likelihood, mll, train_x, train_y, optimizer_type="Adam", epochs=50, lr=0.1):
    """
    Trains the GP model using the specified optimizer strategy.

    Args:
        model: The GPyTorch model.
        likelihood: The GPyTorch likelihood.
        mll: The Marginal Log Likelihood (Loss function).
        train_x: Training inputs (Tensor).
        train_y: Training targets (Tensor).
        optimizer_type (str): 'Adam', 'SGD', 'LBFGS', etc.
        epochs (int): Number of training iterations (ignored for LBFGS).
        lr (float): Learning rate (ignored for LBFGS).
    """
    
    # Set model to training mode
    model.train()
    likelihood.train()

    # --- 1. LBFGS - via Scipy ---
    if optimizer_type == "LBFGS":
        # Botorch provides a robust wrapper for LBFGS on GPyTorch MLLs
        fit_gpytorch_mll_scipy(mll)
        return

    # --- 2. Vanilla PyTorch Optimizers ---
    
    # 1. Initialize the optimizer
    if optimizer_type == "Adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_type == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_type == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown Optimizer: {optimizer_type}")

    # 2. Training Loop
    # Note: Botorch also has 'fit_gpytorch_mll_torch' which does this loop automatically,
    # but we implement the loop manually here to satisfy the project requirement 
    # of "benchmarking vanilla parameter optimizers".
    
    for i in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(train_x)
        
        # Calculate Loss - Negative Log Likelihood
        loss = -mll(output, train_y)
        
        loss.backward()
        optimizer.step()