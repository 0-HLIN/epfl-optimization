# EPFL - OptML - CS439 - Project

## About the project
We implemented Three-Split BCD algorithm introduced in the paper [Global Convergence of Block Coordinate Descent in Deep Learning](https://arxiv.org/abs/1803.00225) by Zeng et al. 
We generalize the algorithm into mini-batch version consisting of inner and outer loop.
Through iterative testing and balancing adjustments of hyperparameters such as batchsize and the number of inner/outer iterations, we sought to examine their influence on the training outcomes. 
The minibatch variant attains similar test accuracy while substantially reducing CUDA memory usage. 
Experimental observations further elucidate the convergence characteristics of the original three-split BCD method. 


## File Structure

    .
    ├── notebooks
    |   ├── bcd.ipynb       # Demo of some experiments
    |   ├── plot.ipynb      # Demo for reproducing plots from logs
    ├── src
    |   ├── bcd.py          # Source code of our BCD implementation
    |   ├── gd.py           # Runner of stardard gradient descent methods
    |   ├── run_bcd.py      # Script of BCD training process
    |   ├── run_gd.py       # Script of (standard) GD training process
    |   ├── utils.py
    ├── results             # Logs of all conducted experiments
    ├── images              # Some typical running plots, produced from logs
    

## Results

The datasets is mnist data, flattened in to 
- x_train of shape (60_000, 28x28), y_train of shape (60_000,)
- x_test of shape (10_000, 28x28), y_test of shape (10_000,)

All BCD use deep network with 3 hidden layers each contains 1500 neurons, the output layer is of size 10. 

### BCD vs Stadard SGD

| Models | CPU | GPU Allocated | GPU Cached | Training time | Final test acc | iters before achieving 0.94 acc | Number of data samples before 0.94 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BCD full batch | 2386.0 - 562.4 | 2090.3 | 5128.0 | ~76  s   | 0.946 | 15    | 9.0e5 |
| BCD b4096      | 2553.2 - 562.5 | 166.0  | 794.0  | ~117 s   | 0.948 | 13x12 | 6.4e5 |
| SGD momentum 0.9 | 2255.6 - 537.9 | 0.7  | 2.0    | ~58  s   | 0.945 | 41    | 2.4e6 |
| RMSProp        | 2258.7 - 539.5   | 0.7  | 2.0    | ~61  s   | 0.965 | 11    | 1.1e5 |
| Adam           | 2257.2 - 539.5   | 0.9  | 2.0     | ~69 s   | 0.957 | 23    | 2.3e5 |

| Train Acc | Test Acc |
|:---:|:---:|
| ![Training acc](images/compare_acc_train.png) |![Test acc](images/compare_acc_test.png) |
| ![Training acc](images/compare_acc_train_with_mb.png) | ![Test acc](images/compare_acc_test_with_mb.png) |


**Notes**
- Memory usage in MB, CPU memory indicates usage after and before training. 
- Training time does not include time consumed for validation during training.
- full batch runs 50 iterations each passes through all 60_000 training data, achieves test acc of 0.94 after 15 epochs.
- BCD b4096 runs 50x12 iterations, each passes through 4096 training data, ahieves test acc of 0.94 after 13x12 iterations. 
- SGD momentum uses momentum coefficient 0.9, lr=1e-1
- RMSprop uses lr=1e-3
- Adam uses lr=3e-4

### Effect of `niter_outer` and `niter_inner` on training

The adapted algorithm:
```python
def train(niter_outer, niter_inner):
    for iter in range(niter_outer):
        x_train, y_train = sample_batch(...)
        forward_pass(x_train)
        for k in range(niter_inner):
            backward_pass(x_train, y_train, ...)
```

| batch size | `niter_outer` | `niter_inner` | acc train* | acc test |
| --- | --- | --- | ---   | ---   |
| 128 | 50  | 60  | 1     | 0.889 |
| 128 | 100 | 30  | 1     | 0.915 |
| 128 | 200 | 15  | 0.961 | 0.933 | 
| 128 | 300 | 10  | 0.959 | 0.931 | 
| 256 | 10  | 150 | 1     | 0.808 | 
| 256 | 50  | 30  | 1     | 0.919 | 
| 256 | 100 | 15  | 0.965 | 0.936 | 
| 256 | 150 | 10  | 0.945 | 0.932 | 
| 256 | 300 | 5   | 0.92  | 0.921 |
| 512 | 3   | 250 | 1     | 0.812 | 
| 512 | 30  | 25  | 0.994 | 0.923 | 
| 1024| 5   | 75  | 1     | 0.889 | 
| 1024| 15  | 25  | 0.98  | 0.932 | 
| 1024| 25  | 15  | 0.966 | 0.934 | 
| 1024| 75  | 5   | 0.917 | 0.919 | 
| 2048| 6   | 30  | 0.985 | 0.934 | 
| 2048| 9   | 20  | 0.971 | 0.936 | 
| 2048| 12  | 15  | 0.950 | 0.937 | 
| 4096| 6   | 15  | 0.956 | 0.937 |
| 4098| 5   | 18  | 0.957 | 0.935 |


**Notes**
- *acc train is the accuracy of the last inner loop iteration, measured on the batch of data used for the inner loop, thus this value could be very noisy due to little data in the mini-batch. acc train is 1 means that model perfectly classifies that mini-batch of data. 
- acc test is the accuracy of the last inner loop iteration, measured on the whole test data (10_000 samples). 

### Inner and Outer Iteration Balance

Setting:
- Experiments with fixed `niter_outer` x `niter_inner` x `batch_size`. 
- For a fixed `batch_size`, different values of `niter_inner` and `niter_outer` have been experimented, though the product of two is fixed.

Observation:
- Too large `niter_inner` leads to overfitting of the model, 
- However the learning will be significantly slower with too small `niter_inner`
- During inner iterations within an outer iteration, the test acc first increase than decrease.

| Batch size | `niter_inner` | `niter_outer` | Results |
|:---:|:---:|:---:|:---:|
|128  |50   |60   | ![bcd_b128_o50_i60_outer](images/bcd_b128_o50_i60_outer.png)  ![bcd_b128_o50_i60_inner](images/bcd_b128_o50_i60_inner.png) |
|128  |200  |15   | ![bcd_b128_o200_i15_outer](images/bcd_b128_o200_i15_outer.png) ![bcd_b128_o200_i15_inner](images/bcd_b128_o200_i15_inner.png) |
|256  |50   |30   | ![bcd_b256_o50_i30_outer](images/bcd_b256_o50_i30_outer.png) ![bcd_b256_o50_i30_inner](images/bcd_b256_o50_i30_inner.png) |
|256  |100  |15   | ![bcd_b256_o100_i15_outer](images/bcd_b256_o100_i15_outer.png) ![bcd_b256_o100_i15_inner](images/bcd_b256_o100_i15_inner.png) |
|1024 |5    |75   | ![bcd_b1024_o5_i75_outer](images/bcd_b1024_o5_i75_outer.png) ![bcd_b1024_o5_i75_inner](images/bcd_b1024_o5_i75_inner.png) |
|1024 |25   |15   | ![bcd_b1024_o25_i15_outer](images/bcd_b1024_o25_i15_outer.png) ![bcd_b1024_o25_i15_inner](images/bcd_b1024_o25_i15_inner.png) |
|1024 |75   |5    | ![bcd_b1024_o75_i5_outer](images/bcd_b1024_o75_i5_outer.png) ![bcd_b1024_o75_i5_inner](images/bcd_b1024_o75_i5_inner.png) |

