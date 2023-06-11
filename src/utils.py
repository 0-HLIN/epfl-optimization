import numpy as np
import psutil
import ray
import os
import time

@ray.remote
class ParameterServer(object):
    def __init__(self, lr, asynchronous, dim=None):
        self.x = np.zeros(dim)
        self.lr = lr
        self.asynchronous = asynchronous

    def apply_gradients(self, grad, *gradients):
        if self.asynchronous:
            self.x -= self.lr * grad
        else:
            summed_gradients = np.sum(gradients, axis=0)
            self.x -= self.lr * summed_gradients
        return self.x

    def get_x(self):
        return self.x
    
    def update_lr(self, lr_coef_mul=1, lr_new=None):
        if lr_new is not None:
            self.lr = lr_new
        else:
            self.lr *= lr_coef_mul
        
    def get_hyperparams(self):
        return self.lr, self.asynchronous
    
@ray.remote
class PSMomentum(ParameterServer):
    def __init__(self, lr, asynchronous, gamma, dim=None):
        super().__init__(lr, asynchronous)
        self.v = np.zeros(dim)
        self.gamma = gamma
        
    def apply_gradients(self, grad, *gradients):
        if self.asynchronous:
            self.v = self.gamma * self.v + self.lr * grad
            self.x -= self.v
        else:
            summed_gradients = np.sum(gradients, axis=0)
            self.v = self.gamma * self.v + self.lr * summed_gradients
            self.x -= self.v
        return self.x



@ray.remote
class DataWorker(object):
    """
    The class for an individual Ray worker.
    Arguments:
        lr (float): the stepsize to be used at initialization
        label (int, optional): batch size for sampling gradients (default: 1)
        seed (int, optional): random seed to generate random variables for reproducibility (default: 0)
        bad_worker (bool, optional): if True, the worker will be forced to be slower than others (default: False)
    """
    def __init__(self, lr, batch_size=1, seed=0, bad_worker=False):
        self.lr = lr
        self.batch_size = batch_size
        self.bad_worker = bad_worker
        self.rng = np.random.default_rng(seed)

    def compute_gradients(self, x):
        t0 = time.perf_counter()
        if self.batch_size is None:
            grad = grad_func(x)
        elif self.batch_size == 1:
            grad = sgrad_func(self.rng, x)
        else:
            grad = batch_grad_func(self.rng, x, self.batch_size)
        if self.bad_worker:
            dt = time.perf_counter() - t0
            time.sleep(100 * dt)
        return grad
    
    def update_lr(self, lr_coef_mul=1, lr_new=None):
        if lr_new is not None:
            self.lr = lr_new
        else:
            self.lr *= lr_coef_mul
        
    def get_hyperparams(self):
        return self.lr, self.batch_size
    
    def get_lr(self):
        return self.lr
    
