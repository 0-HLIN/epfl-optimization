import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


def plot_results(results, xaxis='epoch'):
    loss_tr, loss_te = results['loss_tr'], results['loss_te']
    acc_tr, acc_te = results['acc_tr'], results['acc_te']
    times = results['times']
    niter=len(loss_tr)

    # plot loss and accuracy side by side
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(np.arange(1,niter+1), loss_tr, label='train')
    plt.plot(np.arange(1,niter+1), loss_te, label='test')
    plt.xlabel(xaxis)
    plt.title('loss')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(np.arange(1,niter+1), acc_tr, label='train')
    plt.plot(np.arange(1,niter+1), acc_te, label='test')
    plt.xlabel(xaxis)
    plt.title('accuracy')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(np.arange(1,niter+1), times, label='time')
    plt.axhline(np.mean(times), color='r', linestyle='dashed', label='average time')
    plt.xlabel(xaxis)
    plt.title('time')
    plt.legend()

def plot_results_bcd(results, xlabel='epoch'):
    loss1, loss2 = results['loss1'], results['loss2']
    accuracy_train, accuracy_test = results['acc_tr'], results['acc_te']

    niter=len(loss1)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(np.arange(1,niter+1), loss2, label='loss2')
    plt.title('training loss')
    plt.xlabel(xlabel)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(1,niter+1), accuracy_train, label='train')
    plt.plot(np.arange(1,niter+1), accuracy_test, label='test')
    plt.title('accuracy')
    plt.xlabel(xlabel)
    plt.legend()


def load_mnist_data():
    # Convert to tensor and scale to [0, 1]
    ts = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    mnist_trainset = datasets.MNIST('../data', train=True, download=True, transform=ts)
    mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=ts)
    N = len(mnist_trainset)
    N_test = len(mnist_testset)

    x_train = torch.stack([data[0] for data in mnist_trainset][:N]).reshape(len(mnist_trainset), -1)
    y_train = torch.LongTensor([data[1] for data in mnist_trainset][:N])
    x_test = torch.stack([data[0] for data in mnist_testset][:N_test]).reshape(len(mnist_testset), -1)
    y_test = torch.LongTensor([data[1] for data in mnist_testset][:N_test])

    print("Data loaded:")
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)

    return x_train, y_train, x_test, y_test



def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"CPU Memory usage: {process.memory_info().rss / 1024 ** 2: .1f} MB")

def print_gpu_memory_usage():
    device = torch.device('cuda')
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 2   # cached for PyTorch < 1.0
    print(f"GPU Memory Allocated: {allocated:.1f} MB")
    print(f"GPU Memory Cached: {reserved:.1f} MB")

def print_memory():
    print_memory_usage()
    print_gpu_memory_usage()