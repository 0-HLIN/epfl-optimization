
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time


def relu_prox(a, b, gamma, d, N, device='cpu'):
    val = torch.empty(d,N, device=device)
    x = (a+gamma*b)/(1+gamma)
    y = torch.min(b,torch.zeros(d,N, device=device))

    val = torch.where(a+gamma*b < 0, y, torch.zeros(d,N, device=device))
    val = torch.where(((a+gamma*b >= 0) & (b >=0)) | ((a*(gamma-np.sqrt(gamma*(gamma+1))) <= gamma*b) & (b < 0)), x, val)
    val = torch.where((-a <= gamma*b) & (gamma*b <= a*(gamma-np.sqrt(gamma*(gamma+1)))), b, val)
    return val


class ThreeSplitBCD:
    def __init__(self, data_train, data_test, K, batch_size=128, device='cpu', hidden_sizes=None, depth=3, 
                 activation=nn.ReLU(), activation_prox=relu_prox,
                 seed=42):
        """
        To reproduce the results in the paper, set batch_size to # of training samples.

        Args:
            hidden_sizes: a list of hidden layer sizes
            depth: the number of hidden layers
            K: the number of classes
        """
        self.K = K
        self.batch_size = batch_size
        self.x_train, self.y_train = data_train
        self.x_test, self.y_test = data_test
        self.device = device
        self.activation = activation
        self.activation_prox = activation_prox

        num_features = self.x_train.shape[1]
        if hidden_sizes is None:
            self.layer_sizes = [1500] * depth
        else:
            self.depth = len(hidden_sizes)
            self.layer_sizes = hidden_sizes
        self.layer_sizes = [num_features] + self.layer_sizes + [K]

        # Initialize of parameters
        self.params = []
        for l in range(len(self.layer_sizes) - 1):
            d,d_next = self.layer_sizes[l], self.layer_sizes[l+1]
            W = 0.01*torch.randn(d_next, d, device=device)
            b = 0.1*torch.ones(d_next, 1, device=device)
            self.params.append((W,b))

        self.gammas = [1] * len(self.params)
        self.alphas = [5] * (2 * len(self.params))
        self.rhos = [1] * len(self.params)
    
    def train(self, niter=50):
        # Preprocess data
        # sample batch from training data
        idx_tr = np.random.choice(self.x_train.shape[0], self.batch_size, replace=False)
        x_train, y_train = self.x_train[idx_tr,:], self.y_train[idx_tr]
        x_test, y_test = self.x_test, self.y_test

        x_train, y_train, y_one_hot = self.preprocess_data(x_train, y_train)
        x_test, y_test, _ = self.preprocess_data(x_test, y_test)

        # Forward pass to initialize the neural network states: U,V
        self.forward_pass(x_train)

        # Training logs
        loss1 = np.empty(niter)
        loss2 = np.empty(niter)
        accuracy_train = np.empty(niter)
        accuracy_test = np.empty(niter)
        time1 = np.empty(niter)

        # Backward pass to update the parameters
        for k in range(niter):
            start = time.time()
            self.backward_pass(x_train, y_train, y_one_hot)
            end = time.time()
            time1[k] = end - start

            # Compute loss and accuracy
            l1,l2,acc_tr,acc_te = self.compute_metrcs(x_train, y_train, y_one_hot, x_test, y_test)
            loss1[k] = l1
            loss2[k] = l2
            accuracy_train[k] = acc_tr
            accuracy_test[k] = acc_te

            # print results
            print('Epoch', k + 1, '/', niter, '\n', 
                '-', 'time:', time1[k], '-', 'sq_loss:', loss1[k], '-', 'tot_loss:', loss2[k], 
                '-', 'acc:', accuracy_train[k], '-', 'val_acc:', accuracy_test[k])
            
        return {
            'loss1': loss1,
            'loss2': loss2,
            'acc_tr': accuracy_train,
            'acc_te': accuracy_test,
            'time': time1
        }


    def forward_pass(self, x_train):
        """
        Initialization of the neural network states (output of each layer), corresponding to U,V in the paper.
        Args:
            x_train: tensor of shape (# of features, # of samples)
        """
        self.nn_states = []
        _, N = x_train.shape
        V = x_train
        for W,b in self.params[:-1]:
            U = torch.addmm(b.repeat(1,N), W, V)
            V = self.activation(U)
            self.nn_states.append((U,V))
        W,b = self.params[-1]
        U = torch.addmm(b.repeat(1,N), W, V)
        self.nn_states.append((U, U))

    def backward_pass(self, x_train, y_train, y_one_hot):
        """ this method correspond to one iteration of the inner loop in the paper """
        N = x_train.shape[1]

        for l in range(len(self.params)-1, -1, -1):

            # Retrive values from cache
            d = self.layer_sizes[l+1] # layer_sizes takes into account the input layer
            gamma = self.gammas[l]
            alpha_u, alpha_w = self.alphas[2*l], self.alphas[2*l+1]
            rho = self.rhos[l]
            W,b = self.params[l]
            U,V = self.nn_states[l]
            V_prev = self.nn_states[l-1][1] if l != 0 else x_train

            # Update V, U
            if l != len(self.params)-1:
                rho_next = self.rhos[l+1]
                W_next,b_next = self.params[l+1]
                U_next = self.nn_states[l+1][0]
                V = self.updateV(U,U_next,W_next,b_next,rho_next,gamma)
                U = self.activation_prox(V,(rho*torch.addmm(b.repeat(1,N), W, V_prev) + alpha_u*U)/(rho + alpha_u),(rho + alpha_u)/gamma,d,N, 
                                         device=self.device)
            else: # for the last layer
                V = (y_one_hot + gamma * U + alpha_u * V) / (1 + gamma + alpha_u)
                U = (gamma * V + rho*(torch.mm(W,V_prev) + b.repeat(1,N))) / (gamma + rho)

            # Update W, b
            W,b = self.updateWb(U,V_prev,W,b,alpha_w,rho)

            # Update cache for next iteration
            self.params[l] = (W,b)
            self.nn_states[l] = (U,V)

    def predict(self, x):
        N = x.shape[1]
        for W,b in self.params[:-1]:
            x = self.activation(torch.addmm(b.repeat(1,N), W, x))
        W,b = self.params[-1]
        pred = torch.argmax(torch.addmm(b.repeat(1, N), W, x), dim=0)
        return pred
    
    def compute_metrcs(self, x, y, y_one_hot, x_test=None, y_test=None):
        # Compute loss
        gamma = self.gammas[-1]
        V = self.nn_states[-1][1]
        loss1 = gamma/2 * torch.pow(torch.dist(V,y_one_hot,2),2).cpu().numpy()
        loss2 = loss1.copy()
        for l in range(len(self.params)):
            N = x.shape[1]
            rho = self.rhos[l]
            W,b = self.params[l]
            U = self.nn_states[l][0]
            V_prev = self.nn_states[l-1][1] if l != 0 else x
            loss2 += rho/2 * torch.pow(torch.dist(torch.addmm(b.repeat(1,N), W, V_prev),U,2),2).cpu().numpy()
        # Copmute accuracy
        correct = self.predict(x) == y
        acc = np.mean(correct.cpu().numpy())

        # Compute test accuracy
        if x_test is not None and y_test is not None:
            correct_test = self.predict(x_test) == y_test
            acc_test = np.mean(correct_test.cpu().numpy())
        else:
            acc_test = None
        
        return loss1, loss2, acc, acc_test
    

    def updateV(self, U1,U2,W,b,rho,gamma): 
        _, d = W.size()
        I = torch.eye(d, device=self.device)
        U1 = nn.ReLU()(U1)
        _, col_U2 = U2.size()
        Vstar = torch.mm(torch.inverse(rho*(torch.mm(torch.t(W),W))+gamma*I), rho*torch.mm(torch.t(W),U2-b.repeat(1,col_U2))+gamma*U1)
        return Vstar

    def updateWb(self, U, V, W, b, alpha, rho): 
        d,N = V.size()
        I = torch.eye(d, device=self.device)
        _, col_U = U.size()
        Wstar = torch.mm(alpha*W+rho*torch.mm(U-b.repeat(1,col_U),torch.t(V)),torch.inverse(alpha*I+rho*(torch.mm(V,torch.t(V)))))
        bstar = (alpha*b+rho*torch.sum(U-torch.mm(W,V), dim=1).reshape(b.size()))/(rho*N+alpha)
        return Wstar, bstar
    
    def preprocess_data(self, x, y):
        """ Suppose data x of shape (N, D) and y of shape (N,). 
        """
        N = x.shape[0]
        x = x.to(device=self.device).T
        y_one_hot = torch.zeros(N, self.K).scatter_(1, torch.reshape(y, (N, 1)), 1)
        y_one_hot = torch.t(y_one_hot).to(device=self.device)
        y = y.to(device=self.device)
        return x, y, y_one_hot