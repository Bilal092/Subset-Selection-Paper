import torch
from torch.distributions.multivariate_normal import MultivariateNormal as NMV
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
import torch.nn as nn
import torchvision
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import torch.nn.functional as F

# from subset_select_non_uniform_entropic_FISTA_git import subset_select_non_uniform_FISTA as ss
# from subset_select_ipot_non_uniform_git import subset_select_ipot as ss_ipot

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = "cpu"
    
######################################################################################################################    
# models trained on except separate MNIST case use this model
# class MNIST_classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, dtype=torch.float64)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, dtype=torch.float64)
#         self.fc1 = nn.Linear(256, 128, dtype=torch.float64)
#         self.fc2 = nn.Linear(128, 64, dtype=torch.float64)
#         self.fc3 = nn.Linear(64, 10, dtype = torch.float64)
        

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.softmax(x,dim=-1)
#         return x



# NOisy MNIST is trained 
class MNIST_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, dtype=torch.float64)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=5, dtype=torch.float64)
        self.fc1 = nn.Linear(400, 128, dtype=torch.float64)
        self.fc2 = nn.Linear(128, 64, dtype=torch.float64)
        self.fc3 = nn.Linear(64, 10, dtype = torch.float64)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x_rep = F.relu(self.fc2(x))
        x = self.fc3(x_rep)
        return x, x_rep
    
    
# class FMNIST_classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, dtype=torch.float64)
#         # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=5, dtype=torch.float64)
#         self.fc1 = nn.Linear(400, 128, dtype=torch.float64)
#         self.fc2 = nn.Linear(128, 64, dtype=torch.float64)
#         self.fc3 = nn.Linear(64, 10, dtype = torch.float64)
        

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x_rep = F.relu(self.fc2(x))
#         x = self.fc3(x_rep)
#         return x, x_rep


class FMNIST_classifier(nn.Module):  # extend nn.Module class of nn
    def __init__(self):
        super().__init__()  # super class constructor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5))
        self.batchN1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=128)
        self.batchN2 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):  
        # hidden conv layer
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(input=x, kernel_size=2, stride=2))
        x = self.batchN1(x)

        x = self.conv2(x)
        x = F.relu(F.max_pool2d(input=x, kernel_size=2, stride=2))

        # flatten
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = self.batchN2(x)
        x_rep = self.fc2(x)

        # output
        x = self.fc3(x_rep)

        return x, x_rep
    
##########################################################################################################################   
 
class mushrooms_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=112, out_features=64, dtype=torch.float64, bias=True)
        self.linear2 = nn.Linear(in_features=64, out_features=2, dtype=torch.float64, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
################################################################################################################################

class pageblocks_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=10, out_features=20, dtype=torch.float64, bias=True)
        self.linear2 = nn.Linear(in_features=20, out_features=40, dtype=torch.float64, bias=True)
        self.linear3 = nn.Linear(in_features=40, out_features=20, dtype=torch.float64, bias=True)
        self.linear4 = nn.Linear(in_features=20, out_features=10, dtype=torch.float64, bias=True)
        self.linear5 = nn.Linear(in_features=10, out_features=5, dtype=torch.float64, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x  = (self.linear4(x))
        x =  (self.linear5(x))
        return x
    
######################################################################################################################################

class connect4_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(in_features=126, out_features=256, dtype=torch.float64, bias=True)
        self.linear2 = nn.Linear(in_features=256, out_features=128, dtype=torch.float64, bias=True)
        self.linear3 = nn.Linear(in_features=128, out_features=64, dtype=torch.float64, bias=True)
        self.linear4 = nn.Linear(in_features=64, out_features=3, dtype=torch.float64, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = (self.linear3(x))
        x = self.linear4(x)
        return x
    
######################################################################################################################################

class spambase_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(in_features=57, out_features=256, dtype=torch.float64, bias=True)
        self.linear2 = nn.Linear(in_features=256, out_features=2, dtype=torch.float64, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)


        return x

######################################################################################################################################





    
    



