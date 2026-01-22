import gpytorch
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood, VariationalELBO

def get_mll(mll_type, likelihood, model, train_y=None):
    """
    Factory to return the correct MLL (Marginal Log Likelihood) object.
    Args:mll_type (str): The type of MLL to create. Options: 'ExactMLL', 'LOO', 'ELBO'.
        likelihood (gpytorch.likelihoods.Likelihood): The GP likelihood.
        model (gpytorch.models.GP): The GP model itself.
        train_y (torch.Tensor, optional): The training targets. 
                                          Required for ELBO (to know total data size) 
                                          and helpful for LOO checks.
        
    Returns:gpytorch.mlls.MarginalLogLikelihood: The initialized loss function.
    """
    
    if mll_type == "ExactMLL":
        # Standard Marginal Log Likelihood
        return ExactMarginalLogLikelihood(likelihood, model)
    
    elif mll_type == "LOO":
        # Leave-One-Out Pseudo-Likelihood
        return LeaveOneOutPseudoLikelihood(likelihood, model)
    
    elif mll_type == "ELBO":
        # Variational ELBO (Evidence Lower Bound)
        # We need to know the total number of data points (num_data) for the scaling term
        if train_y is None:
            raise ValueError("VariationalELBO requires 'train_y' to determine num_data.")
        
        return VariationalELBO(likelihood, model, num_data=train_y.size(0))
        
    else:
        raise ValueError(f"Unknown MLL Type: {mll_type}. Supported: ExactMLL, LOO, ELBO")