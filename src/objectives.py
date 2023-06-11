import numpy as np
from abc import ABC, abstractmethod



class Objective:
    def __init__(self, N, D, seed=0):
        """
            N: number of samples
            D: dimension of the data
        """
        # Generate data=(A,b) and x_opt optimal solution on test data
        rng = np.random.default_rng(seed)
        full_A = rng.normal(size=(N+1000,D))
        full_b = rng.normal(size=(N+1000,))
        self.A, self.b = full_A[:N], full_b[:N]
        self.A_test, self.b_test = full_A[N:], full_b[N:]
        self.x_opt = np.linalg.inv(self.A.T @ self.A) @ self.A.T @ self.b
        self.f_min = self.evaluate(self.x_opt)
        self.D = D
    
    # Access to training data (test data is not accessible)
    @property
    def data_train(self):
        return self.A, self.b

    def _eval(self, x, A, b):
        return 0.5 * np.linalg.norm(A @ x - b)**2 / A.shape[0]
    def evaluate(self, x, data=None):
        """ Evaluate on test data """
        if data is None:
            A,b = self.A_test, self.b_test
        else:
            A,b = data
        return self._eval(x, A, b)


class OptimMSE:
    """ Optimizer for MSE loss with penalty. Provude gradient methods
    """
    def __init__(self, data, batch_size=1, reg_coef=1e-4):
        self.A, self.b = data
        self.N, self.D = self.A.shape
        self.B = batch_size
        self.reg_coef = reg_coef
    
    def gradient(self, x, rng):
        idx = rng.choice(self.N, size=self.B, replace=True)
        return (self.A[idx]@x - self.b[idx]) @ self.A[idx] / self.B + self.reg_coef * x

    def grad_func(self, x):
        return (self.A @ x - self.b) @ self.A / self.N + self.reg_coef * x

    def sgrad_func(self, rng, x):
        i = rng.integers(self.N)
        return (self.A[i] @ x - self.b[i]) * self.A[i]

    def batch_grad_func(self, rng, x, batch_size):
        idx = rng.choice(self.N, size=self.B, replace=True)
        return (self.A[idx]@x - self.b[idx]) @ self.A[idx] / self.B + self.reg_coef * x
    def evaluate(self, x):
        assert len(x) == self.D
        return 0.5 * np.mean((self.A @ x - self.b)**2)



