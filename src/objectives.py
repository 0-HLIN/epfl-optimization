import numpy as np
from abc import ABC, abstractmethod


# abstract class for objective functions with abstract methods
# Provide gradient, stochastic gradient, and batch gradient
class Objective(ABC):
    def __init__(self, params):
        pass
    # @abstractmethod
    # def func(self, x):
    #     pass
    @abstractmethod
    def grad_func(self, x):
        pass
    @abstractmethod
    def sgrad_func(self, rng, x):
        pass
    @abstractmethod
    def batch_grad_func(self, rng, x, batch_size):
        pass
    @abstractmethod
    def evaluate(self, x):
        pass


class ObjMSE(object):
    def __init__(self, n_data=10_000, dim=400, noise_scale=1e-5, seed=42):
        rng = np.random.default_rng(seed)
        self.N = n_data
        self.D = dim
        self.noise_scale = noise_scale
        self.A = rng.uniform(size=(n_data, dim)) / np.sqrt(dim)
        x_rand = rng.normal(size=dim)
        self.b = self.A @ x_rand + noise_scale * rng.normal(size=n_data)
        # compute the optimal solution and the minimum objective value for A@x = b
        self.x_opt = np.linalg.lstsq(self.A, self.b, rcond=-1)[0]
        self.f_min = self.evaluate(self.x_opt)

    def grad_func(self, x):
        return (self.A@x - self.b) @ self.A / self.N

    def sgrad_func(self, rng, x):
        i = rng.integers(self.N)
        return (self.A[i] @ x - self.b[i]) * self.A[i]

    def batch_grad_func(self, rng, x, batch_size):
        idx = rng.choice(self.N, size=batch_size, replace=False)
        return (self.A[idx]@x - self.b[idx]) @ self.A[idx] / batch_size
    def evaluate(self, x):
        assert len(x) == self.D
        return 0.5 * np.mean((self.A @ x - self.b)**2)



class ObjRidge(Objective):
    """ Liner objective function with Ridge loss
    """
    def __init__(self, n_data, dim, seed=0, noise_scale=1e-5, batch_size=256, reg_coef=1e-4):
        self.N = n_data
        self.D = dim
        self.noise_scale = noise_scale
        self.reg_coef = reg_coef
        self.rng = np.random.default_rng(seed)
        # Generate data
        self.A = self.rng.uniform(size=(n_data, dim)) / np.sqrt(dim)
        self.x_rand = self.rng.normal(size=dim)
        self.b = self.A @ self.x_rand + self.noise_scale * self.rng.normal(size=n_data)
        # optimal solution TODO: check x_opt
        self.x_opt = np.linalg.inv(self.A.T @ self.A + reg_coef * np.eye(dim)) @ self.A.T @ self.b
        self.f_min = self.evaluate(self.x_opt)

    def grad_func(self, x):
        return (self.A @ x - self.b) @ self.A / self.N + self.reg_coef * x

    def sgrad_func(self, rng, x):
        i = rng.integers(self.N)
        return (self.A[i]@x - self.b[i]) * self.A[i] + self.reg_coef * x

    def batch_grad_func(self, rng, x, batch_size):
        idx = rng.choice(self.N, size=batch_size, replace=False)
        return (self.A[idx]@x - self.b[idx]) @ self.A[idx] / batch_size + self.reg_coef * x

    def evaluate(self, x):
        assert len(x) == self.D
        return 0.5 * np.linalg.norm(self.A @ x - self.b)**2 / self.N + 0.5 * self.reg_coef * np.linalg.norm(x)**2