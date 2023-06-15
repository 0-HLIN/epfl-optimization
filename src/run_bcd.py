

import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import print_memory, load_mnist_data
from bcd import relu_prox, ThreeSplitBCD

print("PyTorch Version:", torch.__version__)
print("GPU is available?", torch.cuda.is_available())

RESULT_DIR = '../results'

#################  Read Data  #################

print("\n######  Loading data  ######")
x_train, y_train, x_test, y_test = load_mnist_data()


#################  Training  #################

print("\n######  Memory before training  ######")
print_memory()

N = x_train.shape[0]
K = 10
result_path = RESULT_DIR + '/bcd_b4096_o50_i12_d3'
batch_size = 4096
num_iter = 50
niter_inner = 12
# result_path = RESULT_DIR + '/bcd_fullbatch_d3'
# batch_size = N # full batch
# num_iter = 1
# niter_inner = 50

start = time.time()
runner = ThreeSplitBCD((x_train, y_train), 
                       (x_test, y_test), 
                       K=K, 
                       batch_size=batch_size, 
                       device='cuda', 
                       hidden_sizes=None, 
                       depth=3,
                       activation=nn.ReLU(),
                       activation_prox=relu_prox,
                       seed=42)

# Experiments for full batch or mini-batch
assert batch_size <= N, "batch_size should be smaller than N"
full_batch = (batch_size == N)
if full_batch:
    results,inner = runner.train(niter=1, niter_inner=niter_inner, log_inner=True, shuffle=False, verbose=True)
else:
    results,inner = runner.train(niter=num_iter, niter_inner=niter_inner, log_inner=True, shuffle=True, verbose=1)


t_train = time.time() - start

print("\n######  Training time (include validation): {:.2f} seconds  ######".format(t_train))
print("\n######  Memory after training  ######")
print_memory()

# Save results
with open(result_path+'.json', 'w') as f:
    json.dump(results, f)
with open(result_path+'_inner.json', 'w') as f:
    json.dump(inner, f)
print("\n######  Results saved to {}  ######".format(result_path))

