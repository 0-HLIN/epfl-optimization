

import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import print_memory, load_mnist_data
from gd import GDStandard

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
batch_size = 64
num_iter = 50 # epoch

# result_path = RESULT_DIR + '/momentum9_b64_h64.json'
# OptimClass = optim.SGD
# optim_args = {'lr': 0.1, 'momentum': 0.9}
# result_path = RESULT_DIR + '/rmsprop_b64_h64_lr1e-3.json'
# OptimClass = optim.RMSprop
# optim_args = {'lr': 1e-3, }
result_path = RESULT_DIR + '/adam_b64_h64_lr3e-4.json'
OptimClass = optim.Adam
optim_args = {'lr': 3e-4, }


start = time.time()
runner = GDStandard((x_train, y_train),
                    (x_test, y_test),
                    K=K,
                    batch_size=batch_size,
                    device='cuda',
                    hidden_sizes=64,
                    OptimClass=OptimClass,
                    optim_args=optim_args,
                    seed=42)

results = runner.train(num_epochs=num_iter)
t_train = time.time() - start


print("\n######  Training time (include validation): {:.2f} seconds  ######".format(t_train))
print("\n######  Memory after training  ######")
print_memory()

# Save results
with open(result_path, 'w') as f:
    json.dump(results, f)
print("\n######  Results saved to {}  ######".format(result_path))

