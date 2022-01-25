import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb


class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H):
        X = torch.mm(X, self.weight) # linear transform : N x w_out
        H = self.norm(H, add=True) # normalize D^(-1)H: N x N 
        return torch.mm(H.t(),X) # N x w_out, why transpose?

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        # D^(-1)H
        # D: degree matrix
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    def forward(self, A, X, target_x, target):
        # A: adjacency matrix: (N,N,K)
        # X: feature matrix for each node: (N,D)
        # target_x: node indices to classify or map or sth
        # target: target value for the nodes in target_x
        A = A.unsqueeze(0).permute(0,3,1,2) # 1 x K x N x N
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)
        for i in range(self.num_channels):
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i])) # take weighted average linearly transformed node representations with metapaths
                # analogous to attention through computed metapaths on linearly transformed node representations
                # X_.shape = N, w_out
            else:
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                X_ = torch.cat((X_,X_tmp), dim=1) # similar to multi attention heads in transformer
        # X_.shape = N, w_out * num_channels 
        X_ = self.linear1(X_) # linear transform
        X_ = F.relu(X_) 
        y = self.linear2(X_[target_x]) # target_x * num_classes
        loss = self.loss(y, target)
        return loss, y, Ws

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels # types of input edges/metapaths, ~ number of input adjacency matrices
        self.out_channels = out_channels # types of output edges/metapaths ~ number of output adjacency matrices
        self.first = first # is first layer
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A) # weighted average of different adjacency matrices
            b = self.conv2(A) # weighted average of different adjacency matrices
            H = torch.bmm(a,b) # meta-path from adjacency matrix a to b
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()] # weights for combinining adjacency matrices, only used for interpretability
        else:
            a = self.conv1(A) # weighted average of different adjancecy matrices
            H = torch.bmm(H_,a) # meta-path from previous layer meta path to a
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()] # weights for combining adjacency matrices
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1)) # 1x1 convolution matrix
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        # A.shape = (1,K,N, N)
        # weight.shape = (O, K, 1, 1)
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1) # combine adjacency matrices, not the sparse version so we don't have temperature
        return A
