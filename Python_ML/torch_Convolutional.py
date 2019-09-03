import torch
import torch.nn as nn
import torch.nn.functional as funcs
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
# from torchvision import datasets, transforms
import h5py
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        super(MyDataset, self).__init__()
        self.store = []
        self.length = len(X)
        for i in range(self.length):
            self.store.append((X[i], Y[i]))
    
    def __getitem__(self, idx):
        return self.store[idx]
    
    def __len__(self):
        return self.length

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, (5, 5), stride = (1, 1), padding = (2, 2))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((8, 8), stride = (8, 8))
        self.conv2 = nn.Conv2d(8, 16, (3, 3), stride = (1, 1), padding = (1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((4, 4), stride = (4, 4))
        self.fc1 = nn.Linear(64, 20)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(20, 6)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(batch_size = 128):
    train_data = h5py.File('/data/tongxueqing/zhaox/train_signs.h5')
    test_data = h5py.File('/data/tongxueqing/zhaox/test_signs.h5')
    X = np.array(train_data["train_set_x"]) / 255
    Y = np.array(train_data["train_set_y"])
    tX = np.array(test_data["test_set_x"]) / 255
    tY = np.array(test_data["test_set_y"])
    train = MyDataset(X.transpose([0, 3, 1, 2]), Y)
    test = MyDataset(tX.transpose([0, 3, 1, 2]), tY)
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader

def init_method(layer):
    if isinstance(layer, nn.Linear):    
        init.xavier_normal_(layer.weight.data)
        init.zeros_(layer.bias.data)

def accuracy(loader, model, name, device = 'cuda'):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype = torch.float)
            y = y.to(device)
            yhat = model(x)
            correct += int(torch.sum(torch.argmax(yhat, axis = 1) == y))
            total += len(y)
    print("Accuracy in %s: %.3f" % (name, 100 * correct / total))

def build_model(loader, num_iterations = 100, learning_rate = 0.0003, device = "cuda"):
    model = Model()
    model = model.apply(init_method)
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr = learning_rate)
    for iteration in range(num_iterations):
        costs = 0
        for x, y in loader:
            x = x.to(device, dtype = torch.float)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            cost = loss(yhat, y)
            cost.backward()
            costs += float(cost)
            optimizer.step()
        if iteration % 100 == 0:
            print("Cost after iteration %d: %.3f" % (iteration, costs))
    return model

def main():
    train_loader, test_loader = load_data()
    model = build_model(train_loader, 3000, 0.0001, "cuda")
    accuracy(train_loader, model, 'train')
    accuracy(test_loader, model, 'test')

main()
# Accuracy in train: 94.907
# Accuracy in test: 92.500