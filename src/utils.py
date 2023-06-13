import numpy as np
import matplotlib.pyplot as plt

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