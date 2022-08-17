import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
import networkx as nx
import torch

class myGraph():
    def __init__(self, dict_data=None, score_data=None):
        self.nodeDict = dict_data
        self.scoreList = score_data
        
        self.degree = None
        self.degreeCoeffient = None

    def nodeProcess(self, files=None, G=None, splitx=' '):
        if files == None and G == None:
            print('here is not any data')
            return 0

        self.nodeDict = {}

        if files != None:
            with open(files, 'r') as f:
                for line in f.readlines():
                    x = line[:-1].split(splitx)
                    x[0] = int(x[0])
                    x[1] = int(x[1])
                    if x[0] not in self.nodeDict.keys():
                        self.nodeDict[x[0]] = []
                    self.nodeDict[x[0]].append(x[1])

                    if x[1] not in self.nodeDict.keys():
                        self.nodeDict[x[1]] = []
                    self.nodeDict[x[1]].append(x[0])

        else:
            edges = list(G.edges)
            for edge in edges:
                if edge[0] not in self.nodeDict.keys():
                    self.nodeDict[edge[0]] = []
                self.nodeDict[edge[0]].append(edge[1])

                if edge[1] not in self.nodeDict.keys():
                    self.nodeDict[edge[1]] = []
                self.nodeDict[edge[1]].append(edge[0])


    def scoreProcess(self, files=None, G_dict=None):
        if files == None and G_dict == None:
            print('here is not any data')
            return 0

        if files != None:
            with open(files, 'r') as f:
                lines = f.readlines()
                self.scoreList = [0. for _ in range(len(lines))]
                for line in lines:
                    x = re.split(':|\t| ', line[:-1])
                    self.scoreList[int(x[0])] = float(x[-1])

        else:
            self.scoreList = [0. for _ in range(len(G_dict.keys()))]
            for i, j in G_dict.items():
                self.scoreList[i] = j

    def get_degree(self):
        self.degree = [0 for _ in range(len(self.scoreList))]
        for i, j in self.nodeDict.items():
            self.degree.append(len(j))

    def trans2tensor(self):
        for i, j in self.nodeDict.items():
            self.nodeDict[i] = torch.tensor(j)
        self.scoreList = torch.tensor(self.scoreList)
        self.degree = torch.tensor(self.degree)

    def degree_Coeffient(self):   #得所有點的 degree 係數向量字典
        if self.nodeDict == None:
            print('no graph')
            return 0

        vectors = []
        N = self.scoreList.shape[0]
        for i in range(N):
            vectors.append(self.degree_v(i))
        self.degreeCoeffient = torch.cat(vectors, dim=0)

    def degree_v(self, v):       # 圖得單一 v 點的 neighborhood 係數向量
        if v in list(self.nodeDict.keys()):
            vneighbor = self.nodeDict[v]
            dv = self.degree[v]
            N = self.scoreList.shape[0]
            t = torch.zeros(1, N, dtype=torch.float)
            for i in vneighbor:
                dn = self.nodeDict[i].shape[0]
                t[0, i] = 1 / ((dv + 1) ** (1 / 2) * (dn + 1) ** (1 / 2))
            
            return t

        else:
            N = self.scoreList.shape[0]
            return torch.zeros(1, N, dtype=torch.float)

    def get_G(self, G):
        self.nodeDict = G.nodeDict
        self.scoreList = G.scoreList
        self.degreeCoeffient = G.degreeCoeffient

    # def sub_graph(self, N=5000, r=False):
    #     if r == False:
    #         vectors = list(np.argsort(np.array(self.scoreList))[-N:])
    #     else:
    #         n = len(self.scoreList)
    #         vectors = random.sample([i for i in range(n)], N)

    #     output = myGraph()
    #     output.scoreList = []
    #     for i in vectors:
    #         output.scoreList.append(self.scoreList[i])

    #     output.nodeDict = {}
    #     for i, j in enumerate(vectors):
    #         temp = list(set(vectors) & set(self.nodeDict[j]))
    #         ls = []
    #         for k in temp:
    #             trc = vectors.index(k)
    #             ls.append(trc)
    #         output.nodeDict[i] = ls

    #     return output


# if self.nodeDict == None:
#     print('empty data')
#     return 0

# l = list(self.nodeDict.keys())
# train_temp, self.test = train_test_split(l, test_size=test)
# self.train, self.validation = train_test_split(train_temp, test_size=validation * (1 - test))