import torch
import gpytorch
import numpy as np
from smac.model.abstract_model import AbstractModel
from gpytorch.likelihoods import GaussianLikelihood
from src.mll_factory import get_mll
from src.optimizers import train_gp

# --- 1. Define the GPyTorch Models ---

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # --- FIX for LBFGS/BoTorch ---
    def transform_inputs(self, X):
        """Required by BoTorch optimizers to handle input transforms."""
        return X 

class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- 2. The SMAC Adapter ---

class GPyTorchSurrogate(AbstractModel):
    def __init__(
        self,
        config_space,
        mll_type="ExactMLL",
        optimizer_type="Adam",
        n_epochs=50,
        learning_rate=0.1,
        pca_components=None,
        seed=0
    ):
        super().__init__(config_space, None, pca_components, seed)
        
        self.mll_type = mll_type
        self.optimizer_type = optimizer_type
        self.n_epochs = n_epochs
        self.lr = learning_rate
        
        self.gp = None
        self.likelihood = None
        self._is_trained = False
        
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def _train(self, X: np.ndarray, y: np.ndarray):
        # 1. Normalize Data
        self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0) + 1e-6
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-6
        
        X_norm = (X - self.X_mean) / self.X_std
        y_norm = (y - self.y_mean) / self.y_std
        
        train_x = torch.tensor(X_norm, dtype=torch.float32)
        train_y = torch.tensor(y_norm, dtype=torch.float32).squeeze()

        # 2. Select Model
        self.likelihood = GaussianLikelihood()
        
        if self.mll_type == "ELBO":
            num_inducing = min(50, train_x.size(0))
            indices = torch.randperm(train_x.size(0))[:num_inducing]
            inducing_points = train_x[indices]
            self.gp = VariationalGPModel(inducing_points)
        else:
            self.gp = ExactGPModel(train_x, train_y, self.likelihood)

        # 3. Get MLL & Train
        mll = get_mll(self.mll_type, self.likelihood, self.gp, train_y)

        train_gp(
            model=self.gp,
            likelihood=self.likelihood,
            mll=mll,
            train_x=train_x,
            train_y=train_y,
            optimizer_type=self.optimizer_type,
            epochs=self.n_epochs,
            lr=self.lr
        )
        
        self._is_trained = True
        return self

    def _predict(self, X: np.ndarray, covariance_type: str = "diagonal"):
        if not self._is_trained:
            raise Exception("Model not trained")

        X_norm = (X - self.X_mean) / self.X_std
        test_x = torch.tensor(X_norm, dtype=torch.float32)

        self.gp.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp(test_x))
            mean = observed_pred.mean.numpy()
            var = observed_pred.variance.numpy()

        mean = (mean * self.y_std) + self.y_mean
        var = var * (self.y_std**2)

        if covariance_type == "full":
            return mean.reshape(-1, 1), np.diag(var)
        
        return mean.reshape(-1, 1), var.reshape(-1, 1)