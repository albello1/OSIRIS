import numpy as np
from scipy import linalg
import copy
from scipy.stats import norm
from sklearn.preprocessing import label_binarize
import math
import sys
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_curve, auc, mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import torch
from torch import nn, optim
import pyro.contrib.gp as gp
import torch

import time

class LR_ARD(object):    
    
    def __init__(self):
        pass

    def fit(self, X_tr, y_tr, X_tst, y_tst, alpha_in = 1, gamma_in = 150, landa_in = 1):
        self.X = [torch.tensor(x).float() for x in X_tr]
        self.X_tst = [torch.tensor(xt) for xt in X_tst]
        self.y = torch.tensor(y_tr).float()*2-1
        self.y_tst = torch.tensor(y_tst).float()*2-1
        self.y_mean = 0

        #Inicializamos las variables
        self.s = len(self.X)
        self.N = np.shape(self.X[0])[0]
        val_alpha = torch.tensor(alpha_in)
        # val_gamma = torch.tensor(150)
        val_gamma = torch.tensor(gamma_in)
        self.alpha = []
        for i in range(self.s):
            self.alpha.append(val_alpha)
        self.gamma = []
        for i in range(self.s):
            self.gamma.append(val_gamma)
        # self.alpha = [val_alpha]*self.s
        # self.gamma = [val_gamma]*self.s
        self.landa = torch.ones(self.N, self.N)*landa_in
        #Inicializamos cada peso
        self.pesos = []
        for i in range(self.s):
            #Calculamos la condicion inicial
            D = self.X[i].size(dim = 1)
            a = torch.sqrt(torch.tensor(1/D, dtype = torch.float))
            self.pesos.append(nn.Parameter(torch.ones(D,1, dtype = torch.float)*a))
        params = self.pesos
        self.optimize(params)

        print('Train: ', self.score(self.X, self.y,1))
        print('Test: ', self.score(self.X_tst, self.y_tst,0))
        
    def update_weights(self):
        term_1 = torch.zeros(self.N,1)
        for i in range(self.s):
            term_1 += self.alpha[i]*self.X[i] @ self.pesos[i]
        term_1 = self.y.T @ term_1

        # term_2 = 0
        # for i in range(self.s -1):
        #     for j in np.arange(i+1, self.s):
        #         term_2 += self.landa[i,j]*self.pesos[i].T @ self.X[i].T @ self.X[j] @ self.pesos[j]
        
        # term_3 = 0
        # for i in range(self.s):
        #     term_3 += self.gamma[i]*(torch.abs(self.pesos[i].T @ self.pesos[i]) -1)

        # fn = term_1 + term_2 - term_3
        fn = term_1
        return fn
    
    def optimize(self, param):

        # define optimizer
        optimizer = torch.optim.SGD(param, lr=0.001, maximize=True)
        # optimizer = [torch.optim.SGD(p, lr=0.01) for p in param]

        loss_SGD = []
        n_iter = 100


        for i in range(n_iter):
            print('Iteration: ', i)

            #Calculamos las penalty
            penalty_ort = 0
            for j in range(self.s):
                penalty_ort += self.gamma[j]*(torch.abs(self.pesos[j].T @ self.pesos[j]) -1)
            penalty_reg = 0
            for i1 in range(self.s -1):
                for j1 in np.arange(i1+1, self.s):
                    penalty_reg += self.landa[i1,j1]*self.pesos[i1].T @ self.X[i1].T @ self.X[j1] @ self.pesos[j1]
            loss = self.update_weights()
            # loss = self.update_weights()
            # store loss into list
            loss_SGD.append(loss.item())
            # zeroing gradients after each iteration
            optimizer.zero_grad()
            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            loss.backward()
            # updateing the parameters after each iteration
            optimizer.step()
            # print(self.pesos[0][0])
            # print(self.gamma)
            print('Loss: ', loss_SGD[-1])
    
    def predict(self, X, y_mean):
        Y = np.zeros((X[0].shape[0],1),dtype=float)
        for i in range(self.s):
            Y += (self.alpha[i].detach().numpy()*X[i].detach().numpy() @ self.pesos[i].detach().numpy())
        Y = Y-y_mean
        return (Y>0)*2-1, Y
    def predict_tr(self, X):
        Y = np.zeros((X[0].shape[0],1),dtype=float)
        for i in range(self.s):
            Y += (self.alpha[i].detach().numpy()*X[i].detach().numpy() @ self.pesos[i].detach().numpy())
        self.y_mean = np.mean(Y)
        Y = Y-self.y_mean
        return (Y>0)*2-1
    
    def score(self, X, y, tr):
        if tr == 1:
            y_pred = self.predict_tr(X)
        else:
            y_pred, y_score = self.predict(X, self.y_mean)
        # print(y_pred)
        # print(y)
        return accuracy_score(y_pred, y)  

    def score_tst(self):
        y_pred, y_score = self.predict(self.X_tst, self.y_mean)
        return accuracy_score(y_pred, self.y_tst)                  