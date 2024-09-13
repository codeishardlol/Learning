import torch
from torch import nn
import torch.nn as nn
class try_nnetwork(nn.Module):
    def __init__(self):
        super(try_nnetwork,self).__init__()
        self.layer1 =nn.Linear(784,256)
        self.layer2=nn.Linear(256,10)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        # #self.gemfield = nn.ModuleList([conv1,pool,conv2,layer1,layer2])
        self.syszux = torch.zeros([1, 1])

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.layer(x)
        # x=x.view(-1,28-28)#view 展平
        # x=self.layer1(x)
        # x=torch.relu(x)
        # return self.layer2(x)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.linear(x)
class nnetwork(nn.Module):
    def __init__(self):
        super(nnetwork, self).__init__()
        self.layer1 =nn.Linear(784,256)
        self.layer2=nn.Linear(256,10)

    def forward(self,x):
        x=x.view(-1,28*28)#view 展平
        x=self.layer1(x)
        x=torch.relu(x)
        return self.layer2(x)

