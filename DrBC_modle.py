import torch
import torch.nn as nn
import numpy as np
import os
import random

class DrBC_encoder(nn.Module):
    def __init__(self, Layer=1, device=None):
        super(DrBC_encoder, self).__init__()
        self.L = Layer
        self.W0 = nn.Sequential(nn.Linear(3, 128), nn.ReLU())
        self.GRU = nn.ModuleList([DrBC_GRU(128, 128) for _ in range(Layer)])

        self.device = device

    def forward(self, inithv, graph):
        hvs = []
        hv = self.W0(inithv)             # N * 3 to N * 128
        hvs.append(hv)
        for i in range(self.L):
            hn = self.neighbor_v(graph, hv)
            hv = self.GRU[i](hv, hn)
            hvs.append(hv)
        output = self.maxPool(hvs)
        return output

    def forward_noCoefficent(self, inithv, graph):
        hvs = []
        hv = self.W0(inithv)             # N * 3 to N * 128
        hvs.append(hv)
        for i in range(self.L):
            hn = self.neighbor_v(graph, hv, CoefficentMatrix=False)
            hv = self.GRU[i](hv, hn)
            hvs.append(hv)
        output = self.maxPool(hvs)
        return output

    def neighbor_v(self, graph, h, CoefficentMatrix=True):        #得圖的所有點的 h neighbor
        if CoefficentMatrix:
            hnCoefficent = graph.degreeCoeffient.to(self.device)
            output = hnCoefficent.mm(h)
        else:
            pass
        return output

    def maxPool(self, hvs):
        for i in range(len(hvs)):
            a, b = hvs[i].shape
            hvs[i] = hvs[i].view(1, a, b)

        newh = torch.cat(hvs, dim=0)
        output = newh.max(dim=0)[0]
        return output

class DrBC_GRU(nn.Module):
    def __init__(self, d_in, d_out):
        super(DrBC_GRU, self).__init__()

        self.W_ir = nn.Linear(d_in, 1, bias=False)
        self.W_hr = nn.Linear(d_in, 1, bias=False)
        self.W_iz = nn.Linear(d_in, 1, bias=False)
        self.W_hz = nn.Linear(d_in, 1, bias=False)

        self.W_in = nn.Linear(d_in, d_out, bias=False)
        self.W_hn = nn.Linear(d_in, d_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x, h):
        r = self.sigmoid(self.W_ir(x) + self.W_hr(h))
        z = self.sigmoid(self.W_iz(x) + self.W_hz(h))
        n = self.tanh(self.W_in(x) + r * self.W_hn(h))
        output = (1 - z) * n + z * h
        
        return output

class DrBC_decoder(nn.Module):
    def __init__(self):
        super(DrBC_decoder, self).__init__()

        self.W = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, hvs):
        output = self.W(hvs)
        return output

class DrBC_module(nn.Module):
    def __init__(self, Layer=1, device=None):
        super(DrBC_module, self).__init__()
        self.layer = Layer
        self.device = device
        self.encoder = DrBC_encoder(Layer=self.layer, device=self.device)
        self.decoder = DrBC_decoder()

    def forward(self, graph, CoefficentMatrix=True):
        if CoefficentMatrix:
            inithv = self.nodeInit(graph).to(self.device)
            embedding = self.encoder(inithv, graph)
            output = self.decoder(embedding)
        else:
            inithv = self.nodeInit(graph).to(self.device)
            embedding = self.encoder.forward_noCoefficent(inithv, graph)
            output = self.decoder(embedding)
        return output

    def nodeInit(self, graph):    # 初始化所有點的 (1, dv ** (1/2)/dv, dv ** (1/3)/dv) 向量
        nodes = []
        n = graph.scoreList.shape[0]
        for v in range(n):
            if v in list(graph.nodeDict.keys()):
                degree = graph.degree[v]
                c = torch.tensor((1, degree ** (1/2) / degree, degree ** (1/3) / degree), dtype=torch.float)
            else:
                c = torch.tensor((1, 0, 0), dtype=torch.float)
            nodes.append(c)
        t = torch.cat(nodes, dim=0).view((-1, 3))
        return t    

    # def nodeInit(self, graph):    # 初始化所有點的 (dv, 1, 1) 向量
    #     nodes = []
    #     n = len(graph.scoreList)
    #     for v in range(n):
    #         if v in list(graph.nodeDict.keys()) :
    #             c = torch.tensor((len(graph.nodeDict[v]), 1, 1), dtype=torch.float)
    #         else:
    #             c = torch.tensor((0, 1, 1), dtype=torch.float)
    #         nodes.append(c)
    #     t = torch.cat(nodes, dim=0).view((-1, 3))
    #     return t

class lossf(nn.Module):
    def __init__(self, device=None, testN=300):
        super(lossf, self).__init__()
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.testN = testN

    def forward(self, hvs, scores):
        train_datas, true_datas = self.randomtest(hvs, scores, n=self.testN)
        n = train_datas.shape[1]
        loss = (-1) * (torch.log(train_datas).mm(true_datas.T) + torch.log(1 - train_datas).mm((1 - true_datas).T)) / n ** (1 / 2)
        return loss
    
    def randomtest(self, hvs, scores, n=300):
        true_datas = []
        train_datas = []
        s = [i for i in range(scores.shape[0])]
        for i in range(n):
            k1, k2 = random.sample(s, 2)
            train_datas.append(hvs[k1, 0:1] - hvs[k2, 0:1])
            true_datas.append(scores[k1] - scores[k2])

        true_datas = torch.tensor(true_datas, dtype=torch.float).to(self.device)
        true_datas = self.sigmoid(true_datas).view(1, -1)
        train_datas = torch.cat(train_datas, dim=0)
        train_datas = self.sigmoid(train_datas).view(1, -1)
        return train_datas, true_datas