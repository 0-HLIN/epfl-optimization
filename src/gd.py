"""
Gradient Descent
"""

# Imports
import torch
import torch.nn as nn
import numpy as np



class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.softmax(self.fc4(out))
        return out
    

class GDStandard:
    def __init__(self, data_train, data_test, K, batch_size=128, device='cpu', 
                 hidden_sizes=64, 
                 OptimClass=torch.optim.SGD,
                 lr=1e-2,
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

        # Initialize the model
        self.model = DNN(self.x_train.shape[1], hidden_sizes, self.K)

        # Initialize the optimizer
        self.optimizer = OptimClass(self.model.parameters(), lr=lr)

        # Initialize the loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize the data loaders
        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.x_train, self.y_train), batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.x_test, self.y_test), batch_size=batch_size, shuffle=True)

    def train(self, num_epochs=100, verbose=True): # TODO change to num_iters? to match BCD
        self.model.to(self.device)

        # Training logs
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Test
            test_loss, test_acc = self.test()
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        return {
            "loss_tr": train_losses,
            "loss_te": test_losses,
            "acc_tr": train_accs,
            "acc_te": test_accs
        }
    
    def train_epoch(self):
        self.model.train()
        train_loss = 0
        train_acc = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_acc += (out.argmax(1) == y).sum().item()
        return train_loss / len(self.train_loader.dataset), train_acc / len(self.train_loader.dataset)
    
    def test(self):
        self.model.eval()
        test_loss = 0
        test_acc = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss = self.loss_fn(out, y)
            test_loss += loss.item()
            test_acc += (out.argmax(1) == y).sum().item()
        return test_loss / len(self.test_loader.dataset), test_acc / len(self.test_loader.dataset)
    
    def predict(self, x):
        self.model.eval()
        out = self.model(x)
        return out.argmax(1)
    