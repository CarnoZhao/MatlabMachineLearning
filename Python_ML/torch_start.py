import torch
import torch.nn as nn
import torch.nn.functional as funcs
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

class Model(nn.Module):
    def __init__(self, n1, n2, n3, n4):
        super(Model, self).__init__()
        self.z1 = nn.Linear(n1, n2)
        self.a1 = nn.ReLU()
        self.z2 = nn.Linear(n2, n3)
        self.a2 = nn.ReLU()
        self.z3 = nn.Linear(n3, n4)
    
    def forward(self, x):
        x = self.z1(x)
        x = self.a1(x)
        x = self.z2(x)
        x = self.a2(x)
        x = self.z3(x)
        return x


def load_data(BATCH_SIZE = 1024):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_data = datasets.MNIST(root = '/data/tongxueqing/zhaox/', train = True, transform = trans)
    test_data = datasets.MNIST(root = '/data/tongxueqing/zhaox/', train = False, transform = trans)
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)
    return train_loader, test_loader

def init_method(layer):
    if isinstance(layer, nn.Linear):    
        init.xavier_normal_(layer.weight.data)
        init.zeros_(layer.bias.data)

def accuracy(model, loader, name):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.view(-1, 28 * 28).cuda()
            y = y.cuda()
            yhat = model(x)
            correct += int(torch.sum(torch.argmax(yhat, axis = 1) == y))
            total += len(y)
    print("Accuracy in %s: %.3f%%" % (name, 100 * correct / total))

def build_model(layer_dims = (28 * 28, 20, 15, 10), device = 'cuda', num_iterations = 30, learning_rate = 0.001):
    train_loader = load_data()[0]
    model = Model(*layer_dims)
    model = model.apply(init_method)
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    opt = optim.Adamax(model.parameters(), lr = learning_rate)
    for iteration in range(num_iterations):
        cost_acc = 0
        for x, y in train_loader:
            x = x.view(-1, 28 * 28).to(device)
            y = y.to(device)
            opt.zero_grad()
            yhat = model(x)
            cost = loss(yhat, y)
            cost.backward()
            cost_acc += float(cost)
            opt.step()
        print("Cost after iteration %d: %.3f" % (iteration, cost_acc))
    return model

model = build_model()
train_loader, test_loader = load_data()
accuracy(model, train_loader, 'training set')
accuracy(model, test_loader, 'test set')