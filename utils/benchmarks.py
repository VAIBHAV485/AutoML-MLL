import numpy as np
from ConfigSpace import ConfigurationSpace, Float

class SyntheticBenchmark:
    """Base class for synthetic functions."""
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, cfg, seed=None):  # <--- CRITICAL FIX: Added seed argument
        raise NotImplementedError

    def get_config_space(self) -> ConfigurationSpace:
        raise NotImplementedError

class Ackley(SyntheticBenchmark):
    def __init__(self, dim=5):
        super().__init__()
        self.dim = dim

    def __call__(self, cfg, seed=None):  # <--- CRITICAL FIX: Added seed argument
        # Convert ConfigSpace dictionary to numpy array
        x = np.array([cfg[f'x{i}'] for i in range(self.dim)])
        
        a, b, c = 20, 0.2, 2 * np.pi
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / self.dim))
        term2 = -np.exp(sum2 / self.dim)
        return term1 + term2 + a + np.exp(1)

    def get_config_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        for i in range(self.dim):
            cs.add(Float(f'x{i}', bounds=(-32.768, 32.768)))
        return cs

class Rosenbrock(SyntheticBenchmark):
    def __init__(self, dim=5):
        super().__init__()
        self.dim = dim

    def __call__(self, cfg, seed=None):  # <--- CRITICAL FIX: Added seed argument
        x = np.array([cfg[f'x{i}'] for i in range(self.dim)])
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def get_config_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        for i in range(self.dim):
            cs.add(Float(f'x{i}', bounds=(-5, 10)))
        return cs

class Hartmann6(SyntheticBenchmark):
    def __call__(self, cfg, seed=None):  # <--- CRITICAL FIX: Added seed argument
        x = np.array([cfg[f'x{i}'] for i in range(6)])
        
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = 10**(-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]])
        
        outer = 0
        for i in range(4):
            inner = np.sum(A[i] * (x - P[i])**2)
            outer += alpha[i] * np.exp(-inner)
        
        return -outer

    def get_config_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        for i in range(6):
            cs.add(Float(f'x{i}', bounds=(0, 1)))
        return cs

class Levy(SyntheticBenchmark):
    def __init__(self, dim=5):
        super().__init__()
        self.dim = dim

    def __call__(self, cfg, seed=None):  # <--- CRITICAL FIX: Added seed argument
        x = np.array([cfg[f'x{i}'] for i in range(self.dim)])
        
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        
        sum_term = 0
        for i in range(self.dim - 1):
            sum_term += (w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2)
            
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        
        return term1 + sum_term + term3

    def get_config_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        for i in range(self.dim):
            cs.add(Float(f'x{i}', bounds=(-10, 10)))
        return cs