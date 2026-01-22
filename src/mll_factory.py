import gpytorch
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood, PredictiveLogLikelihood

def get_mll(mll_type, likelihood, model, train_y=None):
    """
    Factory to return the correct MLL (Marginal Log Likelihood) object.
    """
    
    if mll_type == "ExactMLL":
        return ExactMarginalLogLikelihood(likelihood, model)
    
    elif mll_type == "LOO":
        return LeaveOneOutPseudoLikelihood(likelihood, model)
    
    elif mll_type == "PredictiveLL":
        # Predictive Log Likelihood usually requires num_data for scaling
        if train_y is None:
            raise ValueError("PredictiveLL requires 'train_y' to determine num_data.")
        
        return PredictiveLogLikelihood(likelihood, model, num_data=train_y.size(0))
        
    else:
        raise ValueError(f"Unknown MLL Type: {mll_type}. Supported: ExactMLL, LOO, PredictiveLL")