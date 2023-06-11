import time
import numpy as np
import ray

MAX_SEED = 10_000

class ParameterServerBase(object):
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
class PSDefault(ParameterServerBase):
    pass


@ray.remote
class PSMomentum(ParameterServerBase):
    def __init__(self, lr, asynchronous, gamma=0.9, dim=None):
        super().__init__(lr, asynchronous)
        self.v = np.zeros_like(self.x)
        self.gamma = gamma
        
    def apply_gradients(self, grad, *gradients):
        if not self.asynchronous:
            grad = np.sum(gradients, axis=0)
            print(grad.shape)
        self.v = self.gamma * self.v + (1 - self.gamme) * grad
        self.x -= self.lr * self.v
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
class DataWorker(object):
    """
    The class for an individual Ray worker.
    Arguments:
        lr (float): the stepsize to be used at initialization
        label (int, optional): batch size for sampling gradients (default: 1)
        seed (int, optional): random seed to generate random variables for reproducibility (default: 0)
        bad_worker (bool, optional): if True, the worker will be forced to be slower than others (default: False)
    """
    def __init__(self, optim, lr, batch_size=1, seed=0, bad_worker=False):
        self.optim = optim
        self.lr = lr
        self.batch_size = batch_size
        self.bad_worker = bad_worker
        self.rng = np.random.default_rng(seed)

    def compute_gradients(self, x):
        t0 = time.perf_counter()
        grad = self.optim.gradient(x, self.rng)
        if self.bad_worker:
            dt = time.perf_counter() - t0
            time.sleep(100 * dt)
        return grad
        t0 = time.perf_counter()
        if self.batch_size is None:
            grad = self.optim.grad_func(x)
        elif self.batch_size == 1:
            grad = self.optim.sgrad_func(self.rng, x)
        else:
            grad = self.optim.batch_grad_func(self.rng, x, self.batch_size)
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


def run(obj, optim, seed, num_workers, lr, lr_decay=0, iterations=200, asynchronous=True, delay_adaptive=False, it_check=20,
        batch_size=1, one_bad_worker=False,
        PSClass=PSDefault):
    delays_all = []
    worker_updates = [0 for i in range(num_workers)]
    rng = np.random.default_rng(seed)
    seeds_workers = [rng.choice(MAX_SEED, size=1, replace=False)[0] for _ in range(num_workers)]
    ray.init(ignore_reinit_error=True)
    ps = PSClass.remote(lr, asynchronous, dim=obj.D)
    workers = [DataWorker.remote(optim, lr=lr, batch_size=batch_size, seed=seeds_workers[i]) for i in range(num_workers)]

    x = ps.get_x.remote()
    if asynchronous:
        gradients = {}
        worker_last_it = [0 for _ in range(num_workers)]
        worker_id_to_num = {}
        for e, worker in enumerate(workers):
            gradients[worker.compute_gradients.remote(x)] = worker
            worker_id_to_num[worker] = e


    losses = []
    its = []
    ts = []
    delays = []
    t0 = time.perf_counter()
    delay = 0
    trace = []
    grads_per_it = 1 if asynchronous else num_workers

    for it in range(iterations * (num_workers if asynchronous else 1)):
        n_grads = it * grads_per_it
        if asynchronous:
            ready_gradient_list, _ = ray.wait(list(gradients))
            ready_gradient_id = ready_gradient_list[-1]
            worker = gradients.pop(ready_gradient_id)

            # Compute and apply gradients.
            gradients[worker.compute_gradients.remote(x)] = worker
            worker_num = worker_id_to_num[worker]
            delay = it - worker_last_it[worker_num]
            if delay_adaptive:
                lr_new = lr * num_workers / max(num_workers, delay)
                ps.update_lr.remote(lr_new=lr_new)
            x = ps.apply_gradients.remote(grad=ready_gradient_id)
            worker_last_it[worker_num] = it
            worker_updates[worker_num] += 1
        else:
            gradients = [
                worker.compute_gradients.remote(x) for worker in workers
            ]
            # Calculate update after all gradients are available.
            x = ps.apply_gradients.remote(None, *gradients)

        if it % it_check == 0 or (not asynchronous and it % (max(it_check // num_workers, 1)) == 0):
            # Evaluate the current model.
            x = ray.get(ps.get_x.remote())
            trace.append(x.copy())
            its.append(it)
            ts.append(time.perf_counter() - t0)

        lr_new = lr / (1 + lr_decay * n_grads)
        ps.update_lr.remote(lr_new=lr_new)
        t = time.perf_counter()
        if asynchronous:
            delays.append(delay)

    ray.shutdown()
    return np.asarray(its), np.asarray(ts), np.asarray([obj.evaluate(x) for x in trace]), np.asarray(delays)