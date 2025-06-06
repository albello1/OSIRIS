import numpy as np
from scipy import linalg
import copy
from scipy.stats import norm
from sklearn.preprocessing import label_binarize
import math
import sys
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import torch
from torch import nn, optim
import pyro.contrib.gp as gp
from sklearn import metrics

import time

class OSIRIS(object):    
    
    
    def __init__(self, Kc = 3, Kp = 2, prune = 1, fs = 0, SS_sep = 0,  hyper = None, X_init = None, 
                 G_init = None, W_init = [None],Z_init = None, V_init = None, U_init = None, Wm_init = None, alpha_init = None, 
                 tau_init = None, gamma_init = None, area_mask = None, Yy = None, seed_init = 0):
        self.Kc = int(Kc) 
        self.Kp = int(Kp)
        self.prune = int(prune)     
        self.fs = int(fs) 
        self.SS_sep = SS_sep
        self.hyper = hyper 
        self.seed_init = seed_init
        self.X_init = X_init
        self.G_init = G_init
        self.W_init = W_init
        self.Z_init = Z_init
        self.V_init = V_init
        self.U_init = U_init
        self.Wm_init = Wm_init
        self.alpha_init = alpha_init
        self.tau_init = tau_init
        self.gamma_init = gamma_init
        self.Yy = Yy

    def fit(self, *args,**kwargs):
        """Fit model to data.
        
        Parameters
        ----------
        __X: dict, ('data', 'method', 'sparse').
            Dictionary with the information of the input views. Where 'data' 
            stores the matrix with the data. These matrices have size n_samples
            and can have different number of features. If one view has a number
            of samples smaller than the rest, these values are infered assuming
            it is a semisupervised scheme. This dictionary can be built using 
            the function "struct_data".
        __max_iter: int, (default 500),
            Maximum number of iterations done if not converged.
        __conv_crit: float, (default 1e-6).
            Convergence criteria for the lower bound.
        __verbose: bool, (default 0). 
            Whether or not to print all the lower bound updates.
        __Y_tst: dict, (default [None]).
            If specified, it is used as the output view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
        __X_tst: dict, (default [None]).
            If specified, it is used as the input view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
            
        """
        
        self.n = []
        for (m,arg) in enumerate(args):
            if type(arg) == dict:
                self.n.append(int(arg['data'].shape[0]))
            else:
                for (mx,x) in enumerate(arg):
                    self.n.append(int(arg[mx]['data'].shape[0]))
                    
        self.n_max = np.max(self.n)
        
        self.d = []
        self.sparse = []
        self.sparse_fs = []
        self.center = []
        self.method = []
        self.X = []
        self.t = {}
        self.SS = []
        self.SS_mask = {}   
        self.area_mask = []
        self.V = {}
        self.k = {}
        self.sig = {}
        self.deg = {}
        self.sparse_K = {}

        self.m = -1 
        for arg in args:
            if type(arg) == dict:
                self.m += 1
                self.initializeView(arg)
            else:
                for (mx,x) in enumerate(arg):
                    self.m += 1
                    self.initializeView(x)
        self.L = []
        self.mse = []
        self.R2 = []
        self.HL = []
        self.AUC = []
        self.AUC_tr = []
        self.ACC = []
        self.ACC_tr = []

        self.m = self.m+1
        if self.hyper == None:
            self.hyper = HyperParameters(self.m)
            
        if not (None in self.W_init):
            self.Kc = self.W_init[0]['mean'].shape[1]
        self.q_dist = Qdistribution(self.X, self.n, self.n_max, self.d, self.Kc, self.Kp, self.m, self.sparse, self.method, self.SS, 
                                    self.SS_mask, self.area_mask, self.hyper, 
                                    G_init = self.G_init, W_init = self.W_init, Z_init = self.Z_init, V_init = self.V_init, U_init = self.U_init, Wm_init = self.Wm_init, alpha_init = self.alpha_init, 
                                    tau_init = self.tau_init, gamma_init = self.gamma_init, seed_init=self.seed_init)               
        self.fit_iterate(**kwargs)
        
    def initializeView(self, arg):
        if not (None in arg['data']):
            if arg['method'] == 'reg':   #Regression
                if arg['SV'] is None:
                    self.d.append(int(arg['data'].shape[1]))
                else:
                    self.d.append(int(arg['SV'].shape[0]))
            elif arg['method'] == 'cat': #Categorical
                self.d.append(int(len(np.unique(arg['data'][~np.isnan(arg['data'])]))))
            elif arg['method'] == 'mult': #Multilabel
                if len(arg['data'].shape) < 2:
                    arg['data'] = arg['data'][:, np.newaxis]
                self.d.append(int(arg['data'].shape[1]))
        elif not (None in self.W_init):
            self.d.append(self.W_init[self.m]['mean'].shape[0])  
        self.sparse.append(arg['sparse'])
        self.sparse_fs.append(arg['sparse_fs'])
        self.center.append(arg['center'])
        self.method.append(arg['method'])
        self.area_mask.append(arg['mask'])

        mn = np.random.normal(0.0, 1.0, self.n_max * self.d[self.m]).reshape(self.n_max, self.d[self.m])
        info = {
            "mean":     mn,
            "cov":      mn**2 ,
            "prodT":    np.dot(mn.T,mn),
            "LH":       0,
            "Elogp":    0,
            "sumlogdet":    0,
        }
        self.X.append(info) 
        data = copy.deepcopy(arg['data'])
        
        #Kernel 
        if not (arg['SV'] is None):
            V = copy.deepcopy(arg['SV'])
            X = copy.deepcopy(arg['data'])
            self.k[self.m] = copy.deepcopy(arg['kernel'])
            self.sig[self.m] = copy.deepcopy(arg['sig'])
            self.deg[self.m] = arg['deg']
            if self.k[self.m] == 'linear':
                if self.sparse_fs[self.m]:
                    self.sparse_K[self.m] = SparseELBO(X, V, self.sparse_fs[self.m], kernel = self.k[self.m])
                    data = self.sparse_K[self.m].get_params()[0]
                    self.it_fs = 1
                else: 
                    data = np.dot(X, V.T)
            #RBF Kernel
            elif self.k[self.m] == 'rbf': 
                if self.sig[self.m] == 'auto' or self.sparse_fs[self.m]:
                    self.sparse_K[self.m] = SparseELBO(X, V, self.sparse_fs[self.m])
                    data = self.sparse_K[self.m].get_params()[0]
                    self.it_fs = 1
                else:
                    data, self.sig[self.m] = self.rbf_kernel_sig(X, V, sig = self.sig[self.m])
            elif self.k[self.m] == 'poly':
                from sklearn.metrics.pairwise import polynomial_kernel
                data = polynomial_kernel(X, Y = V, degree = self.deg[self.m], gamma=arg['sig'])
            else:
                print('Error, selected kernel doesn\'t exist')
            if self.center[self.m]:
                data = self.center_K(data)
            # self.d[self.m] = V.shape[0]
            self.X[self.m]['cov'] = np.random.normal(0.0, 1.0, self.n_max * self.d[self.m]).reshape(self.n_max, self.d[self.m])**2
            
        #Regression    
        if arg['method'] == 'reg':   
            # self.t.append(np.ones((self.n_max,)).astype(int))
            self.X[self.m]['mean'] = data
            if self.n_max > self.n[self.m]:
                self.X[self.m]['mean'] = np.vstack((self.X[self.m]['mean'],np.NaN * np.ones((self.n_max - self.n[self.m], self.d[self.m]))))
            self.SS_mask[self.m] = np.isnan(self.X[self.m]['mean'])
            #SemiSupervised
            if np.sum(self.SS_mask[self.m]) > 0:   
                self.SS.append(True)
               #If the matrix is preinitialised
                if (self.X_init is None):
                    for d in np.arange(self.d[self.m]):
                        if np.sum(self.SS_mask[self.m][:,d]) == self.SS_mask[self.m].shape[0]:

                            self.X[self.m]['mean'][self.SS_mask[self.m][:,d],d]= np.random.normal(0.0, 1.0, np.sum(self.SS_mask[self.m][:,d]))
                        else:
                            self.X[self.m]['mean'][self.SS_mask[self.m][:,d],d] = np.random.normal(np.nanmean(self.X[self.m]['mean'][:,d],axis=0), np.nanstd(self.X[self.m]['mean'][:,d],axis=0), np.sum(self.SS_mask[self.m][:,d]))
                else:                        
                    self.X[self.m]['mean'][self.SS_mask[self.m]]  = self.X_init[self.m]['mean'][self.SS_mask[self.m]] 
                self.X[self.m]['cov'][~self.SS_mask[self.m]] = np.zeros((np.sum(~self.SS_mask[self.m]),))
            else:
                self.SS.append(False)
                self.X[self.m]['cov'] = np.zeros((self.X[self.m]['mean'].shape))

        #Categorical
        elif arg['method'] == 'cat':
            self.t[self.m] = {
                "mean":     (np.random.randint(self.d[self.m], size=[self.n_max,])).astype(float),
            }
            data = np.squeeze(data)
            if self.n_max > self.n[self.m]:
                data = np.hstack((data, np.NaN * np.ones((self.n_max - self.n[self.m],))))
            self.SS_mask[self.m] = np.isnan(data)
            print(self.SS_mask[self.m])
            if np.sum(self.SS_mask[self.m]) > 0:
                self.SS.append(True)
                self.t[self.m]['mean'] = (np.random.randint(self.d[self.m], size=[self.n_max, ])).astype(float)
                self.t[self.m]['mean'][~self.SS_mask[self.m]]  = (data[~self.SS_mask[self.m]] ).astype(float)
                self.X[self.m]['mean'][~np.repeat(self.SS_mask[self.m][:,np.newaxis], self.d[self.m],axis=1)] = label_binarize(self.t[self.m]['mean'][~self.SS_mask[self.m]], classes = np.arange(self.d[self.m])).astype(float).flatten()
                self.X[self.m]['cov'][~np.repeat(self.SS_mask[self.m][:,np.newaxis], self.d[self.m],axis=1)] = 0
            else:
                self.SS.append(False)
                self.t[self.m]['mean'] = np.copy(data)
                self.t[self.m]['cov'] = np.zeros((self.t[self.m]['mean'].shape))
                self.X[self.m]['mean'] = np.random.normal(0.0, 1.0, self.n_max * self.d[self.m]).reshape(self.n_max, self.d[self.m])
                self.X[self.m]['cov'] = np.abs(np.random.normal(0.0, 1.0, self.n_max * self.d[self.m]).reshape(self.n_max, self.d[self.m]))

        #Multilabel
        elif arg['method'] == 'mult': 
            self.t[self.m] = copy.deepcopy(info)
            if self.n_max > self.n[self.m]:
                data = np.vstack((data,np.NaN * np.ones((self.n_max - self.n[self.m], self.d[self.m]))))
            self.SS_mask[self.m] = np.isnan(data)
            #Initialization of X using t (The values of t are set to 0.95 and 0.05 to avoid computational problems)
            self.X[self.m]['mean'][~self.SS_mask[self.m]] = np.log(np.abs((data[~self.SS_mask[self.m]]).astype(float)-0.05)/(1 - np.abs((data[~self.SS_mask[self.m]]).astype(float)-0.05)))
            self.X[self.m]['cov'][~self.SS_mask[self.m]] = 0
            for d in np.arange(self.d[self.m]):
                self.X[self.m]['mean'][self.SS_mask[self.m][:,d],d] = np.random.normal(np.nanmean(self.X[self.m]['mean'][~self.SS_mask[self.m][:,d],d],axis=0), np.nanstd(self.X[self.m]['mean'][~self.SS_mask[self.m][:,d],d],axis=0), np.sum(self.SS_mask[self.m][:,d]))
                self.X[self.m]['cov'][self.SS_mask[self.m][:,d],d] = np.abs(np.random.normal(np.nanmean(self.X[self.m]['mean'][~self.SS_mask[self.m][:,d],d],axis=0), np.nanstd(self.X[self.m]['mean'][~self.SS_mask[self.m][:,d],d],axis=0), np.sum(self.SS_mask[self.m][:,d])))

            #SemiSupervised
            if np.sum(self.SS_mask[self.m]) > 0:   
                self.SS.append(True)
                for d in np.arange(self.d[self.m]):
                    self.t[self.m]['mean'][self.SS_mask[self.m][:,d],d] = np.random.normal(np.nanmean(self.t[self.m]['mean'][:,d],axis=0), np.nanstd(self.t[self.m]['mean'][:,d],axis=0), np.sum(self.SS_mask[self.m][:,d]))

                self.t[self.m]['mean'][~self.SS_mask[self.m]]  = (data[~self.SS_mask[self.m]] ).astype(float)
                self.t[self.m]['cov'][~self.SS_mask[self.m]]  = 0

                # Multilabel semisupervised independent
                m_prev = int(np.copy(self.m))
                nans_t = np.sum(np.isnan(data[:self.n[self.m],:]), axis=0)
                if any(nans_t != 0) and self.SS_sep:
                    a = 1 if np.sum(nans_t != 0) != 0 else 0 # If there is no view without nan we keep the first in the original view
                    for v in np.arange(a, self.d[self.m])[nans_t[a:] != 0]:
                        self.m += 1
                        self.sparse.append(arg['sparse'])
                        self.method.append(arg['method'])
                        self.n.append(int(data.shape[0]))
                        self.d.append(int(1))
                        self.SS.append(True)
                        
                        self.X.append(copy.deepcopy(info))
                        self.t.append(copy.deepcopy(self.t[m_prev]))
                        # self.t[self.m]['data'] = self.t[m_prev]['data'][:,v,np.newaxis]
                        self.t[self.m]['mean'] = self.t[m_prev]['mean'][:,v,np.newaxis]
                        self.t[self.m]['cov'] = self.t[m_prev]['cov'][:,v,np.newaxis]
                        # self.X[self.m]['data'] = self.X[m_prev]['data'][:,v,np.newaxis]
                        self.X[self.m]['mean'] = self.X[m_prev]['mean'][:,v,np.newaxis]
                        self.X[self.m]['cov'] = self.X[m_prev]['cov'][:,v,np.newaxis]
                        self.SS_mask[self.m] = np.isnan(data)

                    if a: 
                        self.d[m_prev] = int(1)
                        # self.t[m_prev]['data'] = self.t[m_prev]['data'][:,0,np.newaxis]
                        self.t[m_prev]['mean'] = self.t[m_prev]['mean'][:,0,np.newaxis]
                        self.t[m_prev]['cov'] = self.t[m_prev]['cov'][:,0,np.newaxis]
                        # self.X[m_prev]['data'] = self.X[m_prev]['data'][:,0,np.newaxis]
                        self.X[m_prev]['mean'] = self.X[m_prev]['mean'][:,0,np.newaxis]
                        self.X[m_prev]['cov'] = self.X[m_prev]['cov'][:,0,np.newaxis]
                        self.SS_mask[m_prev] = self.SS_mask[m_prev][:,0,np.newaxis]
                    else:     
                        self.d[m_prev] = int(np.sum(nans_t == 0))
                        # self.t[m_prev]['data'] = self.t[m_prev]['data'][:,nans_t == 0]
                        self.t[m_prev]['mean'] = self.t[m_prev]['mean'][:,nans_t == 0]
                        self.t[m_prev]['cov'] = self.t[m_prev]['cov'][:,nans_t == 0]
                        # self.X[m_prev]['data'] = self.X[m_prev]['data'][:,nans_t == 0]
                        self.X[m_prev]['mean'] = self.X[m_prev]['mean'][:,nans_t == 0]
                        self.X[m_prev]['cov'] = self.X[m_prev]['cov'][:,nans_t == 0]
                        self.SS_mask[m_prev] = self.SS_mask[m_prev][:,nans_t == 0]
                    
                #If the matrix is preinitialised
                if not(self.X_init is None): 
                    #If only the supervised part of the matrix is preinitialised
                    if self.X_init[self.m]['mean'].shape[0]<self.n_max: 
                        self.X[self.m]['mean'][~self.SS_mask[self.m]]  = self.X_init[self.m]['mean'][~self.SS_mask[self.m]] 
                        self.X[self.m]['cov'][~self.SS_mask[self.m]]  = self.X_init[self.m]['cov'][~self.SS_mask[self.m]] 
                    #If all the matrix is preinitialised          
                    else: 
                        self.X[self.m]['mean'][~self.SS_mask[self.m]]  = self.X_init[self.m]['mean'][~self.SS_mask[self.m]] 
                        self.X[self.m]['cov'][~self.SS_mask[self.m]]  = self.X_init[self.m]['cov'][~self.SS_mask[self.m]] 
            else:
                self.SS.append(False)
                self.t[self.m]['mean'] = data.astype(float)
                self.t[self.m]['cov'] = np.zeros(self.t[self.m]['mean'].shape)
                if not(self.X_init is None):
                    if np.max(self.X_init[self.m]['mean']) == 1 and np.min(self.X_init[self.m]['mean']) == 0:
                        self.X[self.m]['mean'] = (2*self.X_init[self.m]['mean']-1).astype(float) #We cast the variables to a float in case the values used
                    else:
                        self.X[self.m]['mean'] = self.X_init[self.m]['mean'].astype(float) #We cast the variable to a float in case the values used
                    self.X[self.m]['cov'] = self.X_init[self.m]['cov'].astype(float) #We cast the variable to a float in case the values used
                    
            
    def fit_iterate(self, max_iter = 250, pruning_crit = 1e-1, tol = 1e-3, feat_crit = 1e-3, perc = False, verbose = 0, Y_tst = [None], 
                    X_tst = [None], X_tr = [None], HL = 0, AUC = 0, ACC= 0, mse = 0, R2 = 0, m_in = [0], m_out = 1):
        """Iterate to fit model to data.
        
        Parameters
        ----------
        __max_iter: int, (default 500),
            Maximum number of iterations done if not converged.
        __conv_crit: float, (default 1e-6).
            Convergence criteria for the lower bound.
        __verbose: bool, (default 0). 
            Whether or not to print all the lower bound updates.
        __Y_tst: dict, (default [None]).
            If specified, it is used as the output view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
        __X_tst: dict, (default [None]).
            If specified, it is used as the input view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
        """
        
        verboseprint = print if verbose else lambda *a, **k: None
        q = self.q_dist
        for i in range(max_iter):            
            # Update the variables of the model
            self.update(Y_tst, X_tst, X_tr, HL, AUC, ACC, mse, R2, m_in, m_out)

            # Pruning if specified after each iteration
            if self.prune:
                # self.depruning(1e-15)
                self.L.append(self.update_bound())
                self.pruning(pruning_crit)
                if q.Kc == 0:
                    print('\nThere are no representative latent factors, no structure found in the data.')
                    return
            else:
                self.L.append(self.update_bound())   
            #Feature selection if specified after each iteration
            if any(self.sparse):
                if self.fs:
                    self.feature_selection(feat_crit, perc)
                    for m in np.arange(self.m):
                        if q.d[m] == 0:
                            print('\nThere are no representative features.')
                            return
            if verbose:
                verboseprint('\rIteration %d Lower Bound %.1f K %4d' %(len(self.L),self.L[-1], q.Kc), end='\r', flush=True)

        verboseprint('')
        
    def rbf_kernel_sig(self, X1, X2, sig=0):
        """RBF Kernel.
            
        Calculates the RBF Kernel between the two different matrices. If a sig
        is not given (sig = 0) we calculate the value of sig.
        
        """
        size1 = X1.shape[0];
        size2 = X2.shape[0];
        if X1.ndim == 1:
            X1 = X1[:,np.newaxis]
            X2 = X2[:,np.newaxis]
        G = (X1* X1).sum(axis=1)
        H = (X2* X2).sum(axis=1)
        Q = np.tile(G, [size2,1]).T
        R = np.tile(H, [size1,1])
        KK = np.dot(X1,X2.T)
        dist = (Q + R - 2*KK)
        if sig == 0:  # Then, we calculate its value
            aux = (dist-np.tril(dist)).reshape(size1*size2,1)
            sig = np.sqrt(0.5*np.mean(aux[np.where(aux>0)]))             
        K = np.exp(-dist/sig**2);
        return K, sig
    
    def center_K(self, K):
        """Center a kernel matrix K, i.e., removes the data mean in the feature space
        Args:
            K: kernel matrix
        """
            
        size_1,size_2 = K.shape;
        D1 = K.sum(axis=0)/size_1
        D2 = K.sum(axis=1)/size_2
        E = D2.sum(axis=0)/size_1
        K_n = K + np.tile(E,[size_1,size_2]) - np.tile(D1,[size_1,1]) - np.tile(D2,[size_2,1]).T
        return K_n

    def pruning(self, pruning_crit):
        """Pruning of the latent variables.
            
        Checks the values of the projection matrices W and keeps the latent 
        variables if there is no relevant value for any feature. Updates the 
        dimensions of all the model variables and the value of Kc.
        
        """
        
        q = self.q_dist
        fact_sel = []
        for m in np.arange(self.m-1):
            fact_sel = fact_sel + np.where(np.any(abs(q.W[m]['mean'])>pruning_crit*np.max(abs(q.W[m]['mean'])), axis=0))[0].tolist()
            # fact_sel = fact_sel + np.where(np.any(abs(q.W[m]['mean'])>pruning_crit, axis=0))[0].tolist()
        fact_sel = fact_sel + np.where(np.any(abs(q.Wm['mean'])>pruning_crit*np.max(abs(q.Wm['mean'])), axis=0))[0].tolist()
        # fact_sel = fact_sel + np.where(np.any(abs(q.Wm['mean'])>pruning_crit, axis=0))[0].tolist()
        fact_sel = np.unique(fact_sel).astype(int)
        # Pruning Z
        q.G['mean'] = q.G['mean'][:,fact_sel]
        q.G['cov'] = q.G['cov'][fact_sel,:][:,fact_sel]
        q.G['prodT'] = q.G['prodT'][fact_sel,:][:,fact_sel]            
         # Pruning W and alpha
        for m in np.arange(self.m-1):
            q.W[m]['mean'] = q.W[m]['mean'][:,fact_sel]
            q.W[m]['cov'] = q.W[m]['cov'][fact_sel,:][:,fact_sel]
            q.W[m]['prodT'] = q.W[m]['prodT'][fact_sel,:][:,fact_sel]   
            q.alpha[m]['b'] = q.alpha[m]['b'][fact_sel]
        q.alpha[-1]['b'] = q.alpha[-1]['b'][fact_sel]
        q.Wm['mean'] = q.Wm['mean'][:,fact_sel]
        q.Wm['cov'] = q.Wm['cov'][fact_sel,:][:,fact_sel]
        q.Wm['prodT'] = q.Wm['prodT'][fact_sel,:][:,fact_sel] 
        q.Kc = len(fact_sel)
        # print('Kc: ', q.Kc)    
        
    def depruning(self, pruning_crit):
        """Pruning of the latent variables.
            
        Checks the values of the projection matrices W and keeps the latent 
        variables if there is no relevant value for any feature. Updates the 
        dimensions of all the model variables and the value of Kc.
        
        """
        
        q = self.q_dist
        K_prune = self.Kc - q.Kc
        q.G['mean'] = np.hstack((q.G['mean'], pruning_crit*np.ones((self.n_max, K_prune))))
        q.G['cov'] = np.vstack((np.hstack((q.G['cov'], pruning_crit*np.ones((q.Kc, K_prune)))), pruning_crit*np.ones((K_prune, self.Kc))))
        q.G['prodT'] = np.vstack((np.hstack((q.G['prodT'], pruning_crit*np.ones((q.Kc, K_prune)))), pruning_crit*np.ones((K_prune, self.Kc))))
         # Pruning W and alpha
        for m in np.arange(self.m-1):
            q.W[m]['mean'] = np.hstack((q.W[m]['mean'], pruning_crit*np.ones((self.d[m], K_prune))))
            q.W[m]['cov'] = np.vstack((np.hstack((q.W[m]['cov'], pruning_crit*np.ones((q.Kc, K_prune)))), pruning_crit*np.ones((K_prune, self.Kc))))
            q.W[m]['prodT'] = np.vstack((np.hstack((q.W[m]['prodT'], pruning_crit*np.ones((q.Kc, K_prune)))),pruning_crit*np.ones((K_prune, self.Kc))))
            q.alpha[m]['b'] = np.hstack((q.alpha[m]['b'], pruning_crit*np.ones((K_prune,))))
        q.alpha[-1]['b'] = np.hstack((q.alpha[-1]['b'], pruning_crit*np.ones((K_prune,))))
        q.Wm['mean'] = np.hstack((q.Wm['mean'], pruning_crit*np.ones((np.shape(self.X[-1]['mean'])[1], K_prune))))
        q.Wm['cov'] = np.vstack((np.hstack((q.Wm['cov'], pruning_crit*np.ones((q.Kc, K_prune)))), pruning_crit*np.ones((K_prune, self.Kc))))
        q.Wm['prodT'] = np.vstack((np.hstack((q.Wm['prodT'], pruning_crit*np.ones((q.Kc, K_prune)))),pruning_crit*np.ones((K_prune, self.Kc))))
            
    def feature_selection(self, feat_crit, perc = False):
        """Feature selection.
            
        Checks the values of the projection matrices W and keeps the features
        if there is no relevant value for any feature. Updates the 
        dimensions of all the model variables and the value of Kc.
        
        Parameters
        ----------
        __feat_crit: float.
            Indicates the feature selection criteria to follow.
        __perc: bool.
            indicates whether the value specified is a threshold or a percentage of features.
            By default it is set to work with a threshold.
        """
        
        q = self.q_dist
        feat_sel = {}
        if perc:
            for m in np.arange(self.m):
                if self.sparse[m]:
                    pos = np.argsort(q.gamma_mean(m))#[::-1]
                    feat_sel[m] = pos[:int(round(self.d[m]*feat_crit))]
            
        else:            
            for m in np.arange(self.m):
                feat_sel[m] = []
                if self.sparse[m]:
                    for d in np.arange(self.d[m]):
                        if any(abs(q.W[m]['mean'][d,:])<feat_crit):
                            feat_sel[m] = np.append(feat_sel[m],d).astype(int)
                            
        # FS W and gamma
        for m in np.arange(self.m):
            if self.sparse[m]:
                self.X[m]['mean'] = self.X[m]['mean'][:,feat_sel[m]]
                self.X[m]['cov'] = self.X[m]['cov'][:,feat_sel[m]]
                if self.SS[m]:
                    self.SS_mask[m] = self.SS_mask[m][:,feat_sel[m]]
                    q.XS[m]['mean'] = q.XS[m]['mean'][:,feat_sel[m]]
                    q.XS[m]['cov'] = q.XS[m]['cov'][:,feat_sel[m]]
                q.W[m]['mean'] = q.W[m]['mean'][feat_sel[m],:]
                q.gamma[m]['b'] = q.gamma[m]['b'][feat_sel[m]]
                q.b[m]['mean'] = q.b[m]['mean'][:,feat_sel[m]]
                q.d[m] = len(feat_sel[m])
        
    def struct_data(self, X, method, sparse = 0, V = None, kernel = None, sig = 0, sparse_fs = 0, center = 1, mask = None, deg = None):
        """Fit model to data.
        
        Parameters
        ----------
        __X: dict, ('data', 'method', 'sparse').
            Dictionary with the information of the input views. Where 'data' 
            stores the matrix with the data. These matrices have size n_samples
            and can have different number of features. If one view has a number
            of samples smaller than the rest, these values are infered assuming
            it is a semisupervised scheme. This dictionary can be built using 
            the function "struct_data".
        __method: char.
            Indicates which type of vraible this is among these:
                'reg'  - regression, floats (shape = [n_samples, n_features]).
                'cat'  - categorical, integers (shape = [n_samples,])
                'mult' - multilabel, one-hot encoding (shape = [n_samples, n_targets])
            
        __sparse: bool, (default 0).
            Indicates if the variable wants to have sparsity in its features 
            or not.
            
        """
        if not (V is None):
            if (kernel is None):
                kernel = 'rbf'
            else:
                kernel = kernel.lower()
            
        X = {"data": X,
        "sparse": sparse,
        "sparse_fs": sparse_fs,
        "method": method,
        "mask": mask,
        "SV": V,
        "kernel": kernel,
        "sig": sig,
        "center": center,
        "deg": deg
        }
        
        if mask is not None and not sparse:
            print('The given mask will not be used as sparsity hasn\'t been selected.')
        # if V is not None:
        #     print('Working on the dual space.')
        return X

    def calcAUC(self, Y_pred, Y_tst):
        n_classes = Y_pred.shape[1]
        
        # Compute ROC curve and ROC area for each class    
        fpr = dict()
        tpr = dict()
        roc_auc = np.zeros((n_classes,1))
        for i in np.arange(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_tst[:,i], Y_pred[:,i]/n_classes)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        return roc_auc.flatten()

    def return_elbo(self):
        return self.L

    def compute_predictions(self, X_tst, m_in=[0], m_out=1, tr=0):
        
        if None in X_tst:
            X_tst = self.X[m_in[0]]     
        n_tst = self.n_max - self.n[m_out]
        if self.method[m_out] == 'reg':
            if self.SS[m_out]:
                Y_pred = self.X[m_out]['mean'][-n_tst:,:]#.reshape(self.n_max - self.n[m_out], self.d[m_out])
            else:
                [Y_pred, var] = self.predict(m_in, m_out, X_tst)
                
        elif self.method[m_out] == 'cat':
            if self.SS[m_out]:
                Y_pred = self.q_dist.tc[m_out][-n_tst:,:] #self.t[m_out]['mean'][-n_tst:,]
            else:
                Y_pred = self.predict(m_in, m_out, X_tst)
                
        elif self.method[m_out] == 'mult':
            if self.SS[m_out]:
#                Y_pred = self.t[m_out]['mean'][self.SS_mask[m_out]].reshape(self.n_max - self.n[m_out], self.d[m_out])
                if tr == 1:
                    Y_pred = self.q_dist.tS[m_out]['mean'][:self.n[m_out],:]
                else:
                    Y_pred = self.t[m_out]['mean'][-n_tst:,:]
            else:
                Y_pred = self.predict(m_in, m_out, X_tst)
        return Y_pred

    def compute_mse(self, Y_tst, X_tst, m_in=[0], m_out=1):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out)
        if self.method[m_out] == 'cat':
            Y_pred = np.argmax()
        d = (Y_tst['data'] - Y_pred).ravel()
        return Y_tst['data'].shape[0]**-1 * d.dot(d)
    
    def compute_R2(self, Y_tst, X_tst, m_in=[0], m_out=1):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out)
        return r2_score(Y_tst['data'], Y_pred[-Y_tst['data'].shape[0]:,:], multioutput = 'uniform_average')
    
    def compute_HL(self, Y_tst, X_tst, m_in=[0], m_out=1):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out)
        if self.method[m_out] == 'cat':
            Y_pred = label_binarize(Y_pred, classes = np.arange(self.d[m_out]))
            Y_tst_bin = label_binarize(Y_tst['data'], classes = np.arange(self.d[m_out]))  
        elif self.method[m_out] in {'reg', 'mult'}:
            Y_pred = (Y_pred > 0.5).astype(int)
            Y_tst_bin = np.copy(Y_tst['data'])
        return hamming_loss(Y_tst_bin.astype(float), Y_pred)
    
    def compute_AUC(self, Y_tst, X_tst, m_in=[0], m_out=1, tr=0):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out, tr=tr)
        if self.method[m_out] == 'cat':
            # Y_pred = label_binarize(Y_pred, classes = np.arange(self.d[m_out]))
            Y_tst_bin = label_binarize(Y_tst['data'], classes = np.arange(self.d[m_out])) 
        else:
            Y_tst_bin = np.copy(Y_tst['data'])
        if self.method[m_out] in {'reg', 'mult'} and Y_tst['data'].shape[1] != self.d[m_out]:
            Y_pred = np.zeros_like(Y_tst['data']).astype(float)
            Y_pred[:,0] = self.t[1]['mean'][(self.n_max - Y_tst['data'].shape[0]):,:].flatten()
            for i in np.arange(1,Y_tst['data'].shape[1]):
                Y_pred[:,i] = self.t[i+1]['mean'][(self.n_max - Y_tst['data'].shape[0]):,:].flatten()
        p_class = np.sum(Y_tst_bin,axis=0)/np.sum(Y_tst_bin)
        return np.sum(self.calcAUC(Y_pred, Y_tst_bin)*p_class)
    
    def mult2lbl(self, Y):
        if (len(Y.shape) < 2) | (Y.shape[1] == 1):
            return (Y > 0.5).astype(float).flatten()
        else:
            return np.argmax(Y, axis=1)

    def compute_ACC(self, Y_tst, X_tst, m_in=[0], m_out=1):
        if not(type(X_tst) == dict):
            m_in = np.arange(self.m-1)
            m_out = self.m-1
        Y_pred = self.compute_predictions(X_tst, m_in=m_in, m_out=m_out)
        if self.method[m_out] == 'reg':
            Y_pred = self.mult2lbl(Y_pred)
            Y_real = self.mult2lbl(Y_tst['data'])
        if self.method[m_out] == 'cat':
            Y_real = Y_tst['data'].flatten()
            Y_pred = (np.ones((Y_pred.shape[0], self.d[m_out])) * np.unique(Y_tst['data']))[label_binarize(np.argmax(abs(Y_pred),axis=1), classes = np.arange(self.d[m_out])).astype(bool)]
        if self.method[m_out] == 'mult':
            Y_pred = self.mult2lbl(Y_pred)
            Y_real = self.mult2lbl(Y_tst['data'])
        return accuracy_score(Y_real, Y_pred)
    
    def update(self, Y_tst=[None], X_tst=[None], X_tr=[None], HL=0, AUC=0, ACC=0, mse=0, R2=0, m_in=[0], m_out=1):
        """Update the variables of the model.
        
        This function updates all the variables of the model and stores the 
        lower bound as well as the Hamming Loss or MSE if specified.
        
        Parameters
        ----------
        __verbose: bool, (default 0). 
            Whether or not to print all the lower bound updates.
        __Y_tst: dict, (default [None]).
            If specified, it is used as the output view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
        __X_tst: dict, (default [None]).
            If specified, it is used as the input view to calculate the 
            Hamming Loss. This dictionary can be built using the function 
            "struct_data".
            
        """

        q = self.q_dist
        
        for m in np.arange(self.m-1):  
            self.update_w(m)
            self.update_V(m)

        self.update_Wm()
        self.update_U()

        self.update_G()
        self.update_Z()

        for m in np.arange(self.m):   
            # batch = np.arange(self.n[m])
            # np.random.shuffle(batch)
            #Regression
            if self.method[m] == 'reg':
                if self.SS[m]:
                    # Updating the mean and variance of X2* for the SS case
                    self.update_xs(m)
                    self.X[m]['mean'][self.SS_mask[m]] = q.XS[m]['mean'][self.SS_mask[m]]
                    self.X[m]['cov'][self.SS_mask[m]] = q.XS[m]['cov'][0,0]
                    self.X[m]['prodT'] = np.dot(self.X[m]['mean'].T, self.X[m]['mean']) + np.diag(np.sum(self.X[m]['cov'],axis=0))
                    X_SS = np.copy(self.X[m]['cov']) # We define a copy of X's covariance matrix to set the observed values to 1 in order to calculate the log determinant
                    X_SS[~self.SS_mask[m]] = 1
                    self.X[m]['sumlogdet'] = np.sum(np.log(X_SS))
                    del X_SS
                else:
                    self.X[m]['prodT'] = np.dot(self.X[m]['mean'].T, self.X[m]['mean'])
                #Update of the variable tau
                self.update_tau(m)

            # Categorical
            elif self.method[m] == 'cat': 
                q.tau[m]['a'] = 1
                q.tau[m]['b'] = 1
                self.update_xcat(m)
                if self.SS[m]:
                    self.update_tc(m)
                    self.t[m]['mean'][self.SS_mask[m]] = np.argmax(q.tc[m][self.SS_mask[m]],axis=1)

            # MultiLabel
            elif self.method[m] == 'mult': 
                for i in np.arange(2):
                    self.update_x(m)
                    self.update_xi(m)
                    if self.SS[m]:
                        # Updating the mean and variance of t* for the SS case
                        self.update_t(m)
                        self.t[m]['mean'][self.SS_mask[m]] = q.tS[m]['mean'][self.SS_mask[m]]

            if self.sparse[m]:
                self.update_gamma(m)
            self.update_alpha(m)
            self.update_mu(m)
            self.update_psi(m)


            
            # if self.sparse_fs[m] and len(self.L) < 50:
            if m in self.k.keys() and self.sparse_fs[m] and self.it_fs and len(self.L) > 1:
                var_fs = self.sparse_K[m].get_params()[1]
                self.sparse_K[m].sgd_step(q.G['mean']@q.W[m]['mean'].T, 20)
                if self.SS[m]:
                    print('Semisupervised version not implemented yet')
                else:
                    kernel = self.sparse_K[m].get_params()[0]
                    self.X[m]['mean'] = self.center_K(kernel)
                    if abs(np.mean(self.sparse_K[m].get_params()[1] - var_fs)) < 1e-6 or abs(self.L[-2]) > abs(self.L[-1]) or len(self.L) > 10:
                        self.it_fs = 0

        self.update_eta()
        self.update_tauz()
        if not(None in Y_tst):
            if HL: 
                self.HL.append(self.compute_HL(Y_tst, X_tst, m_in, m_out))
            if AUC:        
                self.AUC.append(self.compute_AUC(Y_tst, X_tst, m_in, m_out))
            if ACC:
                self.ACC.append(self.compute_ACC(Y_tst, X_tst, m_in, m_out))
            if mse:
                self.mse.append(self.compute_mse(Y_tst, X_tst, m_in, m_out))
            if R2:
                self.R2.append(self.compute_R2(Y_tst, X_tst, m_in, m_out))
            
    def myInverse(self,X):
        """Computation of the inverse of a matrix.
        
        This function calculates the inverse of a matrix in an efficient way 
        using the Cholesky decomposition.
        
        Parameters
        ----------
        __A: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        try:
            # L = linalg.pinv(np.linalg.cholesky(X), rcond=1e-10) 
            # return np.dot(L.T,L) 
            return linalg.pinv(X)
        except:
            return np.nan
        
    def sigmoid(self,x):
        """Computation of the sigmoid function.
        
        Parameters
        ----------
        __x: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        return np.exp(-np.log(1 + np.exp(-x)))
        # return 1. / (1 + np.exp(-x))
  
    def lambda_func(self,x):
        """Computation of the lambda function.
        
        This function calculates the lambda function defined in the paper.
        
        Parameters
        ----------
        __x: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        return np.exp(np.log(self.sigmoid(x) - 0.5) - np.log(2*x))
#        return (self.sigmoid(x) - 0.5)/(2*x)
          
    def update_G(self):
        
        q = self.q_dist
        
        aux = np.eye(q.Kc)
        for m in np.arange(self.m-1):
            aux += q.tau_mean(m)*q.W[m]['prodT']
        aux += np.identity(q.Kc) + q.eta_mean()*q.Wm['prodT'] #OJO AQUI EL ORDEN DE LAS TRANSPUESTAS
        G_cov = self.myInverse(aux)
        if not np.any(np.isnan(G_cov)):
            # cov
            q.G['cov'] = G_cov
            # mean
            mn = np.zeros((self.n_max,q.Kc))
            for m in np.arange(self.m-1):
                mn += np.dot(np.subtract(self.X[m]['mean'], q.b[m]['mean']),q.W[m]['mean']) * q.tau_mean(m)
            mn += q.eta_mean()*self.X[-1]['mean'] @ q.Wm['mean'] - q.eta_mean()*q.Z['mean']@q.U['mean'].T@q.Wm['mean'] #OJO AQUI QUE LE HEMOS QUITADO EL SUMATORIO
            q.G['mean'] = np.dot(mn,q.G['cov'])
            # E[Y*Y^T]
            q.G['prodT'] = np.dot(q.G['mean'].T, q.G['mean']) + self.n_max*q.G['cov'] 
            del aux, G_cov, mn
        else:
            print ('Cov G is not invertible, not updated')
    
    def update_w(self, m):
        
        q = self.q_dist
        
        if self.sparse[m]:
            from numpy import sqrt, divide, dot, diag
            q.W[m]['cov'] = np.zeros((q.Kc,q.Kc))
            q.W[m]['prodT'] = np.zeros((q.Kc,q.Kc))
            q.W[m]['prodTalpha'] = np.zeros((q.d[m],))
            q.W[m]['prodTgamma'] = np.zeros((q.Kc,))
            q.W[m]['sumlogdet'] = 0
            
            A = divide(1.,sqrt(q.alpha_mean(m)))
            U, S, UH = np.linalg.svd(q.tau_mean(m) * A * q.G['prodT'] * A.reshape(-1,1), hermitian=True)
            I = (UH * U.T).sum(-1)  # Only calculates the matrix product diagonal
            AUH = A.reshape(-1,1) * UH.T
            UA  = A * U.T
            
            if self.area_mask[m] is not None:
                for f in np.arange(len(np.unique(self.area_mask[m]))):
                    # w_cov = self.myInverse(diag(q.alpha_mean(m))*q.gamma_mean(m)[f] + q.tau_mean(m) * q.G['prodT'])
                    w_cov = dot(AUH * divide(1.,(I*q.gamma_mean(m)[f] + S)), UA)
                    d = self.area_mask[m] == np.unique(self.area_mask[m])[f]
                    q.W[m]['cov'] += w_cov
                    q.W[m]['mean'][d,:] = np.linalg.multi_dot([(self.X[m]['mean'][:,d] - q.b[m]['mean'][0,d]).T, q.G['mean'] ,w_cov])*q.tau_mean(m)
                    wwT = dot(q.W[m]['mean'][d,:].T, q.W[m]['mean'][d,:]) + w_cov
                    q.W[m]['prodT'] += wwT
                    DwwT = diag(wwT)
                    q.W[m]['prodTgamma'] += q.gamma_mean(m)[f]*DwwT 
                    q.W[m]['prodTalpha'][d] = dot(q.alpha_mean(m),DwwT)
                    q.W[m]['sumlogdet'] += np.linalg.slogdet(w_cov)[1]
            else:
                for d in range(self.d[m]):
                    # w_cov = self.myInverse(diag(q.alpha_mean(m))*q.gamma_mean(m)[d] + q.tau_mean(m) * q.G['prodT'])
                    w_cov = dot(AUH * divide(1.,(I*q.gamma_mean(m)[d] + S)), UA)
                    if not np.any(np.isnan(w_cov)):
                        q.W[m]['cov'] += w_cov
                        q.W[m]['mean'][d,:] = np.linalg.multi_dot([(self.X[m]['mean'][:,d] - q.b[m]['mean'][0,d]).T, q.G['mean'] ,w_cov])*q.tau_mean(m)
                        wwT = dot(q.W[m]['mean'][d,np.newaxis].T, q.W[m]['mean'][d,np.newaxis]) + w_cov
                        q.W[m]['prodT'] += wwT
                        DwwT = diag(wwT)
                        q.W[m]['prodTgamma'] += q.gamma_mean(m)[d]*DwwT 
                        q.W[m]['prodTalpha'][d] = dot(q.alpha_mean(m),DwwT) #¿Se podría quitar el dot?
                        q.W[m]['sumlogdet'] += np.linalg.slogdet(w_cov)[1]
                    else:
                        print ('Cov W is not invertible, not updated')
            del A, U, S, UH, I, AUH, UA, w_cov
        else:
            # cov
            w_cov = self.myInverse(np.diag(q.alpha_mean(m)) + q.tau_mean(m) * q.G['prodT'])
            
            if not np.any(np.isnan(w_cov)):
                q.W[m]['cov'] = w_cov
                # mean
                q.W[m]['mean'] = q.tau_mean(m) * np.linalg.multi_dot([np.subtract(self.X[m]['mean'], q.b[m]['mean']).T,q.G['mean'],q.W[m]['cov']])
                #E[W*W^T]
                q.W[m]['prodT'] = np.dot(q.W[m]['mean'].T, q.W[m]['mean']) + self.d[m]*q.W[m]['cov']
            else:
                print ('Cov W' + str(m) + ' is not invertible, not updated')   
            del w_cov
            
    def update_b(self,m):
        
        q = self.q_dist
        q.b[m]['cov'] = (1 + self.n_max * q.tau_mean(m))**(-1) * np.eye(self.d[m])
        q.b[m]['mean'] = q.tau_mean(m) * np.dot(np.sum(np.subtract(self.X[m]['mean'], np.dot(q.G['mean'], q.W[m]['mean'].T)), axis=0)[np.newaxis,:], q.b[m]['cov'])
        q.b[m]['prodT'] = np.sum(q.b[m]['mean']**2) + self.d[m]*q.b[m]['cov'][0,0]    #mean of a noncentral chi-squared distribution
        
    def diag_sum(self,mat):
        return np.sum(np.diag(mat))
    
    def update_alpha(self,m):
        
        q = self.q_dist
        q.alpha[m]['a'] = (self.hyper.alpha_a[m] + 0.5 * self.d[m])/(self.d[m])
        if m == self.m -1:
            if self.sparse[m]:
                prod = q.Wm['prodTgamma']
            else:
                prod = np.diag(q.Wm['prodT'])
        else:
            if self.sparse[m]:
                prod = q.W[m]['prodTgamma']
            else:
                prod = np.diag(q.W[m]['prodT'])
        q.alpha[m]['b'] = (self.hyper.alpha_b[m] + 0.5 * prod)/(self.d[m])
        
    def update_tau(self,m):
        
        q = self.q_dist
        q.tau[m]['a'] = (self.hyper.tau_a[m] + 0.5 * self.d[m]*self.n_max)/(self.d[m]*self.n_max) 
        q.tau[m]['b'] = (self.hyper.tau_b[m] + 0.5 *(np.trace(self.X[m]['prodT']) + np.trace(np.dot(q.W[m]['prodT'],q.G['prodT'])) 
                                                     - 2 * np.trace(np.linalg.multi_dot([q.W[m]['mean'], q.G['mean'].T,self.X[m]['mean']])) 
                                                     + 2 * np.sum(np.linalg.multi_dot([q.G['mean'], q.W[m]['mean'].T,q.b[m]['mean'].T])) 
                                                     - 2 *np.sum(np.dot(self.X[m]['mean'],q.b[m]['mean'].T)) 
                                                     + self.n_max * q.b[m]['prodT'] ))/(self.d[m]*self.n_max)

    def update_gamma(self,m):
       
        q = self.q_dist     
        q.gamma[m]['a'] = (self.hyper.gamma_a[m] + 0.5 * q.Kc)/q.Kc
        if m == self.m-1:
            q.gamma[m]['b'] = (self.hyper.gamma_b[m] + 0.5 *q.Wm['prodTalpha'])/q.Kc
        else:
            q.gamma[m]['b'] = (self.hyper.gamma_b[m] + 0.5 *q.W[m]['prodTalpha'])/q.Kc
                
    def update_xs(self,m): #Semisupervised
        
        
        q = self.q_dist
        # cov
        q.XS[m]['cov'] = (q.tau_mean(m)**(-1)*np.eye(self.d[m])).astype(float)
        # mean
        q.XS[m]['mean'] = np.add(np.dot(q.G['mean'],q.W[m]['mean'].T), q.b[m]['mean'])
    
    def update_t(self,m): 
        
        
        q = self.q_dist
        # mean
        # print('Max X: ',self.X[m]['mean'].max())
        # print('Min X: ',self.X[m]['mean'].min())
        q.tS[m]['mean'] = self.sigmoid(self.X[m]['mean'])
        # cov
        # q.tS[m]['cov'] = np.exp(np.subtract(np.log(q.tS[m]['mean']), np.log((1 + np.exp(self.X[m]['mean'])))))
        q.tS[m]['cov'] = np.exp(np.subtract(self.X[m]['mean'], 2*np.log((1 + np.exp(self.X[m]['mean'])))))
        # print('Min cov: ',np.min(q.tS[m]['cov']))
        # sum(log(det(X)))
        q.tS[m]['sumlogdet'] = np.sum(np.log(q.tS[m]['cov']))
            
    def update_x(self,m): #Multilabel
        
        
        q = self.q_dist
        # self.X[m]['cov'] = (q.tau_mean(m) + 2*self.lambda_func(q.xi[m]))**(-1)
        self.X[m]['cov'] = (q.eta_mean() + 2*self.lambda_func(q.xi[m]))**(-1)
        concatenados = np.concatenate((np.dot(q.G['mean'], q.Wm['mean'].T),np.dot(q.Z['mean'], q.U['mean'].T)), axis = 1)
        self.X[m]['mean'] = (self.t[m]['mean'] - 0.5 + q.eta_mean()*(np.dot(q.G['mean'], q.Wm['mean'].T) + np.dot(q.Z['mean'], q.U['mean'].T))) * self.X[m]['cov']
        # self.X[m]['mean'] = (self.t[m]['mean'] - 0.5 + q.eta_mean()*(np.dot(q.Z['mean'], q.U['mean'].T))) * self.X[m]['cov']
        # self.X[m]['mean'] = (self.t[m]['mean'] - 0.5 + q.eta_mean()*(np.dot(q.G['mean'], q.Wm['mean'].T))) * self.X[m]['cov']
        self.X[m]['prodT'] = np.dot(self.X[m]['mean'].T, self.X[m]['mean']) + np.diag(np.sum(self.X[m]['cov'],axis=0))
        self.X[m]['sumlogdet'] = np.sum(np.log(self.X[m]['cov']))

    def update_xi(self,m): #Multilabel    
        
        q = self.q_dist
        q.xi[m] = np.sqrt(self.X[m]['cov'] + self.X[m]['mean']**2)

    ######## Predictive
    
    def calculate_w_cov(self, fix, vary):
        return fix*vary
    
    def calculate_w_mean(self, fix, vary):
        return fix @ vary
    
    def take_value_list(self, lista, value):
        return lista[value]
    
    def update_V(self,m):
        q = self.q_dist
        #Computation of covariance
        ################
        fixed = q.tauz_mean()*self.X[m]['prodT']
        w_covs = list(map(self.calculate_w_cov,q.Kp*[np.diagflat(q.mu_mean(m))], list(q.psi_mean(m))))
        w_covs = w_covs + fixed
        w_covs_inv = list(map(self.myInverse, w_covs))
        #######################
        indices_nonan = [index for index, value in enumerate(w_covs_inv) if not isinstance(value, float) or not (value != value)]
        for pos in indices_nonan:
            q.V_cov[m]['mean'][pos] = w_covs_inv[pos]
        #Computation of the mean
        w_m = np.zeros((q.Kp,self.n_max))
        for mo in range(self.m-1):
            w_m += (self.X[mo]['mean'] @ q.V[mo]['mean'].T).T
        w_m = w_m - (self.X[m]['mean'] @ q.V[m]['mean'].T).T
        w_mean_list = list(map(self.calculate_w_mean, q.Kp*[q.tauz_mean()*(q.Z['mean'].T - w_m) @ self.X[m]['mean']], q.V_cov[m]['mean']))
        w_mean_def = list(map(self.take_value_list,w_mean_list, np.arange(0,q.Kp)))
        w_mean_def = np.array(w_mean_def)
        q.V[m]['mean'] = w_mean_def

    def update_Z(self):
        q = self.q_dist

        z_cov = q.tauz_mean()*np.eye(q.Kp) + q.eta_mean()*q.U['prodT']
        z_cov_inv = self.myInverse(z_cov)    
        if not np.any(np.isnan(z_cov_inv)):
            q.Z['cov'] = z_cov_inv
            z_m = np.zeros((self.n_max, q.Kp))
            for m1 in range(self.m-1):
                z_m += self.X[m1]['mean'] @ q.V[m1]['mean'].T
            z_mean = (q.tauz_mean()*z_m + q.eta_mean()*q.X[-1]['mean'] @ q.U['mean'] - q.eta_mean()*q.G['mean']@q.Wm['mean'].T @ q.U['mean']) @ q.Z['cov']
            q.Z['mean'] = z_mean
            q.Z['prodT'] = q.Z['mean'].T @ q.Z['mean'] + q.Z['cov']
        else:
            print ('Cov Z is not invertible, not updated')

        #Seguir por aqui
    
    def update_psi(self,m):
        
        q = self.q_dist
        
        q.psi[m]['a'] = self.d[m]/2 + self.hyper.psi_a[m]
        
        psi_b = np.diag(q.mu_mean(m)*q.V[m]['mean']@q.V[m]['mean'].T) + list(map(self.diag_sum,q.mu_mean(m)*q.V_cov[m]['mean']))
        q.psi[m]['b'] = self.hyper.psi_b[m] + (1/2)*psi_b
        
        
    def update_mu(self,m):
        q = self.q_dist
        
        
        q.mu[m]['a'] = q.Kp/2 + self.hyper.mu_a[m]
        
        mu_b = np.zeros((self.d[m],))
        for d in range(self.d[m]):
            prov = 0
            for k in range(q.Kp):
                prov += q.psi_mean(m)[k]*(q.V[m]['mean'][k,d]*q.V[m]['mean'][k,d] + q.V_cov[m]['mean'][k][d,d])
            mu_b[d] = prov
        q.mu[m]['b'] = self.hyper.mu_b[m] + (1/2)*mu_b
    
    def update_tauz(self):
        q = self.q_dist
        q.tauz['a'] = (self.Kp*self.n_max)/2 + self.hyper.tauz_a

        term1 = np.trace(q.Z['mean'] @ q.Z['mean'].T) + self.n_max*np.trace(q.Z['cov'])
        
        term2_prov = 0
        for m in range(self.m-1):
            term2_prov += (self.X[m]['mean'] @ q.V[m]['mean'].T).T
        term2 = np.trace(q.Z['mean'] @ term2_prov)

        term3_prov = 0
        for m1 in range(self.m-1):
            for m2 in range(self.m-1):
                if m1 != m2:
                    term3_prov += np.trace(self.X[m1]['mean'] @ q.V[m1]['mean'].T @ (self.X[m2]['mean'] @ q.V[m2]['mean'].T).T)
                elif m1 == m2:
                    wtw = np.zeros((self.d[m1], self.d[m1]))
                    for k in range(q.Kp):
                        wtw += q.V[m1]['mean'][np.newaxis,k,:].T @ q.V[m1]['mean'][np.newaxis,k,:] + q.V_cov[m1]['mean'][k]
                    term3_prov += np.trace(self.X[m1]['prodT'] @ wtw)
        term3 = term3_prov
        q.tauz['b'] = self.hyper.tauz_b + 0.5*term1 - term2 + 0.5*term3
    
        
    def update_U(self):
        q = self.q_dist

        U_covs = []
        for c in range(np.shape(self.X[-1]['mean'])[1]):
            U_cov = np.identity(q.Kp) + q.eta_mean()*q.Z['prodT']
            U_covs.append(U_cov)
        U_covs_inv = list(map(self.myInverse, U_covs))
        U_mean = np.zeros((np.shape(self.X[-1]['mean'])[1],q.Kp))
        for c in range(np.shape(self.X[-1]['mean'])[1]):
            U_mean[c,:] = (q.eta_mean()*self.X[-1]['mean'][:,c, np.newaxis].T @ q.Z['mean'] - q.eta_mean()*q.Wm['mean'][np.newaxis,c,:] @ q.G['mean'].T @ q.Z['mean']) @ U_covs_inv[c]
        q.U['mean'] = U_mean
        utu = 0
        for c in range(np.shape(self.X[-1]['mean'])[1]):
            utu += q.U['mean'][np.newaxis,c,:].T @ q.U['mean'][np.newaxis,c,:] + U_covs_inv[c]
        q.U['prodT'] = utu
    
    def update_Wm(self):
        q = self.q_dist
        if self.sparse[-1]:
            Wm_covs = []
            for c in range(np.shape(self.X[-1]['mean'])[1]):
                Wm_cov = q.eta_mean()*q.G['prodT'] + q.gamma_mean(-1)[c]*np.diagflat(q.alpha_mean(-1))
                Wm_covs.append(Wm_cov)
            Wm_covs_inv = list(map(self.myInverse, Wm_covs))
            Wm_mean = np.zeros((np.shape(self.X[-1]['mean'])[1],q.Kc))
            for c in range(np.shape(self.X[-1]['mean'])[1]):
                Wm_mean += (q.eta_mean()*q.X[-1]['mean'][:,c,np.newaxis].T @ q.G['mean'] - q.eta_mean()*q.U['mean'][np.newaxis,c,:] @ q.Z['mean'].T @ q.G['mean'])@Wm_covs_inv[c]
            q.Wm['mean'] = Wm_mean
            wtw = 0
            q.Wm['prodTalpha'] = np.zeros((np.shape(self.X[-1]['mean'])[1],))
            q.Wm['prodTgamma'] = np.zeros((q.Kc,))
            for c in range(np.shape(self.X[-1]['mean'])[1]):
                wtw += q.Wm['mean'][np.newaxis,c,:].T @ q.Wm['mean'][np.newaxis,c,:] + Wm_covs_inv[c]
                DwwT = np.diag(wtw)
                q.Wm['prodTgamma'] += q.gamma_mean(-1)[c]*DwwT 
                q.Wm['prodTalpha'][c] = np.dot(q.alpha_mean(-1),DwwT)
            q.Wm['prodT'] = wtw
        else:
            Wm_cov = q.eta_mean()*q.G['prodT'] + np.diagflat(q.alpha_mean(-1))
            Wm_covs_inv = self.myInverse(Wm_cov)
            Wm_mean = np.zeros((np.shape(self.X[-1]['mean'])[1],q.Kc))
            for c in range(np.shape(self.X[-1]['mean'])[1]):
                Wm_mean += (q.eta_mean()*q.X[-1]['mean'][:,c,np.newaxis].T @ q.G['mean'] - q.eta_mean()*q.U['mean'][np.newaxis,c,:] @ q.Z['mean'].T @ q.G['mean'])@Wm_covs_inv
            q.Wm['mean'] = Wm_mean
            wtw = 0
            for c in range(np.shape(self.X[-1]['mean'])[1]):
                wtw += q.Wm['mean'][np.newaxis,c,:].T @ q.Wm['mean'][np.newaxis,c,:] + Wm_covs_inv[c]
            q.Wm['prodT'] = wtw

    def update_eta(self):
        q = self.q_dist

        q = self.q_dist

        
        q.eta['a'] = (self.hyper.eta_a + 0.5 * np.shape(self.X[-1]['mean'])[1]*self.n_max)
        q.eta['b'] = (self.hyper.eta_b + 0.5 *(np.trace(self.X[-1]['prodT']) + np.trace(np.dot(q.U['prodT'],q.Z['prodT'])) + np.trace(np.dot(q.Wm['prodT'],q.G['prodT']))
                                                     - 2 * np.trace(np.linalg.multi_dot([q.Wm['mean'], q.G['mean'].T,self.X[-1]['mean']])) 
                                                     - 2 * np.trace(np.linalg.multi_dot([q.U['mean'], q.Z['mean'].T,self.X[-1]['mean']]))
                                                     + 2 * np.trace(q.Z['mean'] @ q.U['mean'].T @ q.Wm['mean'] @ q.G['mean'].T) 
                                                     ))




        
        
    def expectation_aprx(self, a, b, c = [None], n_samples = 100, n = None):
        """Calculates the expectation aproximation.
                
        Parameters
        ----------
        __a: float.
            Mean value of the wanted class.
            
        __b: float.
            Mean value of the not wanted classes.
            
        __c: float, (default [None])
            In case there is a pdf in the expectation, this parameter is the one
            used for the mean. N(c - a, 1).
                       
        __n_samples: int, (default 100).
            
        __n: int, (default None).
            

        """

        if n == None:
            n = self.n_max
             
        exp = 0
        for it in np.arange(n_samples):
            u = np.random.normal(0.0, 1.0, n)
            prod = 1
#            prod = 0
            for j in np.arange(np.shape(b)[1]):
                prod = prod * norm.cdf(u + a - b[:,j], 0.0, 1.0) #We calculate the cdf for each class
#                prod = prod + np.log(norm.cdf(u + a - b[:,j], 0.0, 1.0)) #We calculate the cdf for each class
            if not (None in c):
                exp += norm.pdf(u, c - a, 1)*prod
#                exp += np.exp(np.log(norm.pdf(u, c - a, 1)) + prod)
            else:
                exp += prod
#                exp += np.exp(prod)
        return exp/n_samples
    
    def update_xcat(self,m): #Multiclass
        """Updates the variable X.
        
        This function uses the variables of the learnt model to update X of 
        the specified view in the case of a categorical view.
        
        Parameters
        ----------
        __m: int. 
            This value indicates which of the input views is updated.

        """
        
        q = self.q_dist
        y = np.dot(q.G['mean'],q.W[m]['mean'].T) + q.b[m]['mean']
           
        set_classes = np.unique(self.t[m]['mean']).astype(int) 
        t_b = label_binarize(self.t[m]['mean'], classes=set_classes).astype(bool)
        if t_b.shape[1] == 1:
            t_b = np.hstack((~t_b, t_b))              
        y_i = y[t_b]
        y_j = y[~t_b].reshape(self.n_max,self.d[m]-1)

        exp_j = np.zeros((self.n_max,self.d[m]-1))
        for j in np.arange(self.d[m]-1):
            # y_k = y_j[:,np.arange(self.d[m]-1)!=j] #it extracts the mean of the values there are neither i nor j
            exp_j[:,j] = self.expectation_aprx(y_i, y_j[:,np.arange(self.d[m]-1)!=j], c = y_j[:,j])
        # mean
        self.X[m]['mean'][~t_b] = (y_j - (exp_j.T/self.expectation_aprx(y_i, y_j) + 1e-10).T).flatten()
        self.X[m]['mean'][t_b] = y_i + np.sum(y_j - self.X[m]['mean'][~t_b].reshape(self.n_max,self.d[m]-1),axis=1)
        self.X[m]['prodT'] = np.dot(self.X[m]['mean'].T, self.X[m]['mean'])

    def update_tc(self,m): #Semisupervised categorical
        q = self.q_dist
        for i in np.arange(self.d[m]):
            q.tc[m][:,i] = self.expectation_aprx(self.X[m]['mean'][:,np.arange(self.d[m]) == i].flatten(), self.X[m]['mean'][:,np.arange(self.d[m]) != i])
            
    def predict(self, m_in, m_out, *args):
        """Apply the model learned in the training process to new data.
        
        This function uses the variables of the specified views to predict
        the output view.
        
        Parameters
        ----------
        __X: dict, ('data', 'method', 'sparse').
            Dictionary with the information of the input views. Where 'data' 
            stores the matrix with the data. These matrices have size n_samples
            and can have different number of features. If one view has a number
            of samples smaller than the rest, these values are infered assuming
            it is a semisupervised scheme. This dictionary can be built using 
            the function "struct_data".
            
        __m_in: list. 
            This value indicates which of the views are used as input.        
        __m_out: list. 
            This value indicates which of the input views is used as output.
        """
# =============================================================================
#         Hay que modificarlo para que pueda predecir todo lo que quieras a la vez. 
#         Para ello hay que definir un m_vec = [0,1,0,0,1] indicando qué vistas
#         son para calcular la predicción y cuáles para ser predichas.
# =============================================================================

        q = self.q_dist
        
        if type(args[0]) == dict:
            n_pred = np.shape(args[0]['data'])[0] 
        else:
            n_pred = np.shape(args[0][0]['data'])[0] 
        
        aux = np.eye(q.Kc)
        for m in m_in:
            aux += q.tau_mean(m)*np.dot(q.W[m]['mean'].T,q.W[m]['mean'])
        G_cov = self.myInverse(aux)
        
        if not np.any(np.isnan(G_cov)):
            self.G_mean = np.zeros((n_pred,q.Kc))
            for (m,arg) in enumerate(args):
                if not (arg['SV'] is None) and not(arg['data'].shape[1] == arg['SV'].shape[0]):
                    V = copy.deepcopy(arg['SV'])
                    X = copy.deepcopy(arg['data'])
                    k = copy.deepcopy(arg['kernel'])
                    sig = copy.deepcopy(arg['sig'])
                    center = copy.deepcopy(arg['center'])
                    #Feature selection
                    #Lineal Kernel
                    if k == 'linear':
                        arg['data'] = np.dot(X, V.T)
                    #RBF Kernel
                    elif k == 'rbf': 
                        if sig == 'auto':
                            self.sparse_K[m] = SparseELBO(X, V, self.sparse_fs[m])
                            arg['data'], _ = self.sparse_K[m].get_params()[0]
                        else:
                            arg['data'], sig = self.rbf_kernel_sig(X, V, sig = sig)
                    if center:
                        arg['data'] = self.center_K(arg['data'])
                        
                if type(arg) == dict:
                    if arg['method'] == 'cat': #categorical
                        arg['data'] = label_binarize(arg['data'], classes = np.arange(self.d[m_in[m]]))
                    self.G_mean += np.dot(arg['data'],q.W[m_in[m]]['mean']) * q.tau_mean(m_in[m])
                else:
                    for (m,x) in enumerate(arg):
                        if x['method'] == 'cat': #categorical
                            x['data'] = label_binarize(x['data'], classes = np.arange(self.d[m_in[m]]))
                        self.G_mean += np.dot(x['data'],q.W[m_in[m]]['mean']) * q.tau_mean(m_in[m])
                    
            self.G_mean = np.dot(self.G_mean,G_cov)
        else:
            print ('Cov G is not invertible')

        ########################
        
        aux = np.eye(q.Kp)
        for m in m_in:
            aux += np.dot(q.V[m]['mean'],q.V[m]['mean'].T)
        Z_cov = self.myInverse(aux)*q.tauz_mean()
        
        if not np.any(np.isnan(Z_cov)):
            self.Z_mean = np.zeros((n_pred,q.Kp))
            for (m,arg) in enumerate(args):
                if not (arg['SV'] is None) and not(arg['data'].shape[1] == arg['SV'].shape[0]):
                    V = copy.deepcopy(arg['SV'])
                    X = copy.deepcopy(arg['data'])
                    k = copy.deepcopy(arg['kernel'])
                    sig = copy.deepcopy(arg['sig'])
                    center = copy.deepcopy(arg['center'])
                    #Feature selection
                    #Lineal Kernel
                    if k == 'linear':
                        arg['data'] = np.dot(X, V.T)
                    #RBF Kernel
                    elif k == 'rbf': 
                        if sig == 'auto':
                            self.sparse_K[m] = SparseELBO(X, V, self.sparse_fs[m])
                            arg['data'], _ = self.sparse_K[m].get_params()[0]
                        else:
                            arg['data'], sig = self.rbf_kernel_sig(X, V, sig = sig)
                    if center:
                        arg['data'] = self.center_K(arg['data'])
                        
                if type(arg) == dict:
                    if arg['method'] == 'cat': #categorical
                        arg['data'] = label_binarize(arg['data'], classes = np.arange(self.d[m_in[m]]))
                    self.Z_mean += np.dot(arg['data'],q.V[m_in[m]]['mean'].T) * q.tauz_mean()
                else:
                    for (m,x) in enumerate(arg):
                        if x['method'] == 'cat': #categorical
                            x['data'] = label_binarize(x['data'], classes = np.arange(self.d[m_in[m]]))
                        self.Z_mean += np.dot(x['data'],q.V[m_in[m]]['mean']) * q.tauz_mean()
                    
            self.Z_mean = np.dot(self.Z_mean,Z_cov)
        else:
            print ('Cov Z is not invertible')

        ########################

        #Regression
        if self.method[m_out] == 'reg':   
           
            return np.dot(self.G_mean,q.Wm['mean'].T) + np.dot(self.Z_mean,q.U['mean'].T), q.eta_mean()**(-1)*np.eye(self.d[m_out]) + np.linalg.multi_dot([q.W[m_out]['mean'], G_cov, q.W[m_out]['mean'].T]) + q.tauz_mean()**(-1)*np.eye(self.d[m_out]) + np.linalg.multi_dot([q.V[m_out]['mean'], Z_cov, q.V[m_out]['mean'].T])             
        
        elif self.method[m_out] == 'cat': 
            p_t = np.zeros((n_pred,self.d[m_out]))
            y = np.dot(self.G_mean,q.W[m_out]['mean'].T)
            for i in np.arange(self.d[m_out]):
                p_t[:,i] = self.expectation_aprx(y[:,np.arange(self.d[m_out]) == i].flatten(), y[:,np.arange(self.d[m_out]) != i], n = n_pred)
            # return np.argmax(abs(p_t),axis=1)    
            return p_t
         
        #Multilabel
        elif self.method[m_out] == 'mult': 
       
            
            p_t = np.zeros((n_pred,self.d[m_out]))
            #Probability t
            for d in np.arange(np.shape(self.X[-1]['mean'])[1]):
                p_t[:,d] = self.sigmoid((np.dot(self.G_mean, q.Wm['mean'].T) + np.dot(self.Z_mean, q.U['mean'].T))[:,d]*(1+math.pi/8*(q.eta_mean()**(-1)*np.eye(np.shape(self.X[-1]['mean'])[1]) + np.linalg.multi_dot([q.Wm['mean'], G_cov, q.Wm['mean'].T])[d,d] + q.tauz_mean()**(-1)*np.eye(np.shape(self.X[-1]['mean'])[1]) + np.linalg.multi_dot([q.U['mean'], Z_cov, q.U['mean'].T])[d,d]))**(-0.5))

            return p_t
       
    def HGamma(self, a, b):
        """Compute the entropy of a Gamma distribution.

        Parameters
        ----------
        __a: float. 
            The parameter a of a Gamma distribution.
        __b: float. 
            The parameter b of a Gamma distribution.

        """
        
        return -np.log(b + sys.float_info.epsilon)
    
    def HGauss(self, mn, cov, entr):
        """Compute the entropy of a Gamma distribution.
        
        Uses slogdet function to avoid numeric problems. If there is any 
        infinity, doesn't update the entropy.
        
        Parameters
        ----------
        __mean: float. 
            The parameter mean of a Gamma distribution.
        __covariance: float. 
            The parameter covariance of a Gamma distribution.
        __entropy: float.
            The entropy of the previous update. 

        """
        
        H = 0.5*mn.shape[0]*np.linalg.slogdet(cov)[1]
        return self.checkInfinity(H, entr)
        
    def checkInfinity(self, H, entr):
        """Checks if three is any infinity in th entropy.
        
        Goes through the input matrix H and checks if there is any infinity.
        If there is it is not updated, if there isn't it is.
        
        Parameters
        ----------
        __entropy: float.
            The entropy of the previous update. 

        """
        
        if abs(H) == np.inf:
            return entr
        else:
            return H
        
    def update_bound(self):
        """Update the Lower Bound.
        
        Uses the learnt variables of the model to update the lower bound.
        
        """
        
        q = self.q_dist
        
        # Entropy of Z
        q.G['LH'] = self.HGauss(q.G['mean'], q.G['cov'], q.G['LH'])
        for m in np.arange(self.m):
            # Entropy of W
            if self.sparse[m]:
                q.W[m]['LH'] = 0.5*q.W[m]['sumlogdet']
                q.gamma[m]['LH'] = np.sum(self.HGamma(q.gamma[m]['a'], q.gamma[m]['b']))
            else: 
                q.W[m]['LH'] = self.HGauss(q.W[m]['mean'], q.W[m]['cov'], q.W[m]['LH'])
            # Entropy of b
            q.b[m]['LH'] = self.HGauss(q.b[m]['mean'], q.b[m]['cov'], q.b[m]['LH'])
            # Entropy of alpha and tau
            q.alpha[m]['LH'] = np.sum(self.HGamma(q.alpha[m]['a'], q.alpha[m]['b']))
            q.tau[m]['LH'] = np.sum(self.HGamma(q.tau[m]['a'], q.tau[m]['b']))
            # Entropy of X if multilabel
            if self.method[m] == 'mult':
                self.X[m]['LH'] = self.checkInfinity(0.5*self.X[m]['sumlogdet'], self.X[m]['LH'])
            # Entropies if semisupervised 
            if self.SS[m]:
                if self.method[m] == 'reg':
                    q.XS[m]['LH'] = self.checkInfinity(0.5*self.X[m]['sumlogdet'], q.XS[m]['LH'])
                    # q.XS[m]['LH'] = self.HGauss(q.XS[m]['mean'][self.n[m]:,:], q.XS[m]['cov'], q.XS[m]['LH'])
                if self.method[m] == 'mult':
                    q.tS[m]['LH'] = self.checkInfinity(0.5*q.tS[m]['sumlogdet'], q.tS[m]['LH'])

        # Total entropy
        EntropyQ = q.G['LH']
        for m in np.arange(self.m):
            EntropyQ += q.W[m]['LH'] + q.b[m]['LH'] + q.alpha[m]['LH']  + q.tau[m]['LH']
            if self.sparse[m]:
                EntropyQ += q.gamma[m]['LH']
            if self.method[m] == 'mult':
                EntropyQ += self.X[m]['LH']
            if self.SS[m]:
                if self.method[m] == 'reg':
                    EntropyQ += q.XS[m]['LH']
                if self.method[m] == 'mult':
                    EntropyQ += q.tS[m]['LH']
        
        # Calculation of the E[log(p(Theta))]
        q.G['Elogp'] = -0.5*np.trace(q.G['prodT'])
        for m in np.arange(self.m):   
            q.b[m]['Elogp'] = -0.5*q.b[m]['prodT']
            q.tau[m]['ElogpXtau'] = -(0.5*self.n_max * self.d[m] + self.hyper.tau_a[m] -1)* np.log(q.tau[m]['b'] + sys.float_info.epsilon)
            if self.method[m] == 'mult': #MultiLabel
                q.tau[m]['ElogpXtau'] += np.sum(np.log(self.sigmoid(q.xi[m])) + self.X[m]['mean'] * self.t[m]['mean'] - 0.5 * (self.X[m]['mean'] + q.xi[m]))
            if self.sparse[m]: #Even though it sais Walp, it also includes the term related to gamma
                q.alpha[m]['ElogpWalp'] = -(0.5* self.d[m] + self.hyper.alpha_a[m] -1)* np.sum(np.log(q.alpha[m]['b'])) -(0.5* q.Kc + self.hyper.gamma_a[m] -1)* np.sum(np.log(q.gamma[m]['b'])) #- self.hyper.gamma_b[m]*np.sum(q.gamma_mean(m))
            else:                    
                q.alpha[m]['ElogpWalp'] = -(0.5* self.d[m] + self.hyper.alpha_a[m] -1)* np.sum(np.log(q.alpha[m]['b']))


        # Total E[log(p(Theta))]
        ElogP = q.G['Elogp']
        for m in np.arange(self.m):
            ElogP += q.tau[m]['ElogpXtau'] + q.alpha[m]['ElogpWalp'] + q.b[m]['Elogp']
        return ElogP - EntropyQ

##############################################################################
##############################################################################
from torch.nn import Parameter

class LinearARD(nn.Module):

    def __init__(self, input_dim, variance=None, ARD=True):
        super(LinearARD, self).__init__()
        if variance is None:
            variance = torch.Tensor(np.ones((input_dim,)))
        elif variance.shape[0] == 1:
            variance = variance * torch.Tensor(np.ones((input_dim,)))
        elif variance.shape[0] != input_dim:
            raise ValueError("Inputs must have the same number of features.")
        self.ARD = ARD
        self.log_variance = Parameter(variance)

    def _dot_product(self, X, X2=None):
        if X2 is None:
            return torch.mm(X, X)
        else:
            return torch.mm(X, X2.T)

    def forward(self, X, X2):
        if self.ARD:
            rv = torch.sqrt(self.log_variance.exp())
            if X2 is None:
                return torch.mm(X * rv,(X * rv).T)
            else:
                return torch.mm(X * rv, (X2 * rv).T)
        else:
            return self._dot_product(X, X2) * self.log_variance.exp()
        
##############################################################################
##############################################################################

class SparseELBO(nn.Module):

    def __init__(self, X, V, fs, lr=1e-3, kernel='rbf', var = None):
        '''
        This class optimizes the lengthscale of each dimension of the X and V data points
        to give ARD to the system.
        Parameters
        ----------
        X : Numpy Array
            Data array with shape NxD containing the full data points to train.
        V : Numpy Array
            Support vector array with shape N'xD (N'<<N) containing the points 
            to be used as support vectors.
        lr : float, optional
            Learning rate to the Adam optimizer. The default is 1e-3.
        kernel: str, optional
            Type of kernel to use. "linear" and "rbf" are the ones implemented.

        Returns
        -------
        None.
        '''
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.X = torch.from_numpy(X).to(self.device)
        self.V = torch.from_numpy(V).to(self.device)
        self.k_type = kernel
        if fs:
            is_var = False
            var = 1.0
            is_leng = True
            leng = 100.
        else:
            is_var = True
            var = 0.01  
            is_leng = False
            leng = 1.
        
        if var:
            is_var = False
        else:
            var = 0.02*torch.ones([self.X.shape[1],]).to(self.device)
         
            
        if self.k_type == 'rbf':
            self.kernel = gp.kernels.RBF(input_dim=self.X.shape[1], variance=torch.tensor(var),
                lengthscale = leng * torch.ones([self.X.shape[1],]).to(self.device),
                active_dims = range(X.shape[1]))
            self.kernel.lengthscale_unconstrained.requires_grad = is_leng
            self.kernel.variance_unconstrained.requires_grad = is_var
        elif self.k_type == 'linear':
            self.kernel = LinearARD(input_dim = self.X.shape[1], variance = var * torch.ones([self.X.shape[1],]).to(self.device))
            self.kernel.variance_unconstrained.requires_grad = not is_var
        self.K = self.kernel.forward(self.X, self.V)        
        self.opt = optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)

    def forward(self, ZAT):
        '''
        Defines the RBF kernel and calculate the ELBO.
        Parameters
        ----------
        ZAT : numpy array.
            Matrix product of Z@A.T with shape NxN'.

        Returns
        -------
        M : torch tensor.
            ELBO function as a tensor of shape NxN'. Without doing the summation
            on axes N and N'.

        '''
        self.K = self.kernel.forward(self.X, self.V)
        return self.K * ZAT - 0.5 * self.K.pow(2)
    
    def get_params(self):
        '''
        Returns the lengthscale and the variance of the RBF kernel in numpy form

        Returns
        -------
        float32 numpy array
            A row vector of length D with the value of each lengthscale in float32.
        float32
            A float value meaning the variance of the RBF kernel in float32.

        '''
        if self.k_type == 'rbf':
            lengthscale = np.exp(self.kernel.lengthscale_unconstrained.data.cpu().numpy())
            return self.K.data.cpu().numpy(), lengthscale, self.kernel.variance.data.cpu().numpy()
        elif self.k_type == 'linear':
            variance = np.exp(self.kernel.variance_unconstrained.data.cpu().numpy())
            return self.K.data.cpu().numpy(), variance
         
    def sgd_step(self, ZAT, it):
        '''
        Computes "it" steps of the Adam optimizer to optimize our ELBO.
        Parameters
        ----------
        ZAT : numpy array
            Matrix product of Z@A.T with shape NxN'..
        it : integer
            Integer that indicates how many steps of the optimizer has to do.

        Returns
        -------
        None.

        '''
        ZAT = torch.from_numpy(ZAT).to(self.device)
        for i in range(it):
            self.opt.zero_grad()
            self.ELBO_loss = torch.sum(self.forward(ZAT))
            self.ELBO_loss.backward()
            self.opt.step()
            
class HyperParameters(object):
    """ Hyperparameter initialisation.
    
    Parameters
    ----------
    __m : int.
        number of views in the model.
    
    """
    def __init__(self, m_i):
        self.alpha_a = []
        self.alpha_b = []
        self.gamma_a = []
        self.gamma_b = []
        self.psi_a = []
        self.psi_b = []
        self.mu_a = []
        self.mu_b = []
        self.tau_a = []
        self.tau_b = []
        self.eta_a = 1e-14
        self.eta_b = 1e-14
        self.tauz_a = 1e-14
        self.tauz_b = 1e-14

        self.xi = []
        for m in np.arange(m_i): 
    
            
            self.tau_a.append(1e-14)
            self.tau_b.append(1e-14)
            
 
            self.psi_a.append(1e-14)
            self.psi_b.append(1e-14)

            self.mu_a.append(2)
            self.mu_b.append(1)

            #########################

            self.alpha_a.append(2)
            self.alpha_b.append(1)
            

            
            self.gamma_a.append(1e-14)
            self.gamma_b.append(1e-14)


            
class Qdistribution(object):
    """ Hyperparameter initialisation.
    
    Parameters
    ----------
    __m : int.
        number of views in the model.
    
    """
    def __init__(self, X, n, n_max, d, Kc, Kp, m, sparse, method, SS, SS_mask, area_mask, hyper, G_init=None, 
                 W_init=None, Z_init = None, V_init = None, U_init = None, Wm_init = None, alpha_init=None, tau_init=None, gamma_init=None, seed_init = 0):
        self.n = n
        self.n_max = n_max
        self.d = d
        self.Kc = Kc
        self.seed_init = seed_init
        self.Kp = Kp
        self.m = m
        self.sparse = sparse
        self.SS = SS
        self.X = X
        # Initialize some parameters that are constant
        self.alpha = self.qGamma(hyper.alpha_a,hyper.alpha_b,self.m,(self.Kc*np.ones((self.m,))).astype(int)) if alpha_init is None else alpha_init
        self.tau = self.qGamma(hyper.tau_a,hyper.tau_b,self.m,(np.ones((self.m))).astype(int)) if tau_init is None else tau_init
        # We generate gamma for all views
        self.gamma = self.qGamma(hyper.gamma_a,hyper.gamma_b,self.m,self.d, area_mask, self.sparse) if gamma_init is None else gamma_init
        self.eta = self.qGamma_uni(hyper.eta_a,hyper.eta_b,1)
        self.tauz = self.qGamma_uni(hyper.tauz_a,hyper.tauz_b,1)

        self.psi = self.qGamma(hyper.psi_a,hyper.psi_b,self.m,(self.m)*[self.Kp]) 
        self.mu = self.qGamma(hyper.mu_a,hyper.mu_b,self.m,self.d) 
        
        self.xi = [None]*m
        for m in np.arange(self.m):       
            if method[m] == 'mult':
                self.xi[m] = np.sqrt(self.X[m]['cov'] + self.X[m]['mean']**2)
            
        # The remaning parameters at random 
        self.init_rnd(X, method, SS, SS_mask, G_init, W_init, Z_init, V_init, U_init, Wm_init)

    def init_rnd(self, X, method, SS, SS_mask, G_init=None, W_init=None, Z_init = None, V_init = None, U_init = None, Wm_init = None):
        """ Hyperparameter initialisation.
    
        Parameters
        ----------
        __m : int.
            number of views in the model.
            
        """
        np.random.seed(self.seed_init)
        W = [None]*(self.m)
        for m in np.arange(self.m):
            info = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            W[m] = info
        V = [None]*(self.m)
        for m in np.arange(self.m):
            info = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            V[m] = info
        self.V_cov = [None]*(self.m)
        for m in np.arange(self.m):
            info_Vcov = {
                "mean":     [None]*self.Kp,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            self.V_cov[m] = info_Vcov
        self.XS = copy.deepcopy(W)
        self.tS = copy.deepcopy(W)
        self.b = copy.deepcopy(W)
        self.tc = {}
        G = copy.deepcopy(W[0])
        Z = copy.deepcopy(W[0])
        U = copy.deepcopy(W[0])
        Wm = copy.deepcopy(W[0])

        # Initialization of U
        U['mean'] = np.random.normal(0.0, .01, self.d[-1] * self.Kp).reshape(self.d[-1], self.Kp)
        U['cov'] = np.eye(self.d[-1]) #np.dot(self.G['mean'].T,self.G['mean']) 
        U['prodT'] = U['cov'] + self.n_max*U['cov'] #np.dot(self.G['mean'].T, self.G['mean']) + self.n_max*self.G['cov']
        if U_init is None: #If the matrix is not initialised
            self.U = U
        elif U_init['mean'].shape[0]<self.n_max: #If only the supervised part of the matrix is initialised
            self.U = U
            self.U['mean'][:U_init['mean'].shape[0],:] = U_init['mean']
            self.U['cov'] = U_init['cov']
            self.U['prodT'] = U_init['prodT']
        else: #If all the matrix is initialised          
            self.U = U_init
        
        # Initialization of Wm
        Wm['mean'] = np.random.normal(0.0, 1.0, self.d[-1] * self.Kc).reshape(self.d[-1], self.Kc)
        Wm['cov'] = np.eye(self.Kc) #np.dot(self.G['mean'].T,self.G['mean']) 
        Wm['prodT'] = Wm['cov'] + self.n_max*Wm['cov'] #np.dot(self.G['mean'].T, self.G['mean']) + self.n_max*self.G['cov']
        # W[m]['prodTalpha'] = np.zeros((self.d[-1],))
        # W[m]['prodTgamma'] = np.zeros((self.Kc,))
        if Wm_init is None: #If the matrix is not initialised
            self.Wm = Wm
        elif Wm_init['mean'].shape[0]<self.n_max: #If only the supervised part of the matrix is initialised
            self.Wm = Wm
            self.Wm['mean'][:Wm_init['mean'].shape[0],:] = Wm_init['mean']
            self.Wm['cov'] = Wm_init['cov']
            self.Wm['prodT'] = Wm_init['prodT']
        else: #If all the matrix is initialised          
            self.Wm = Wm_init
            
        # Initialization of the generative space matrix G
        G['mean'] = np.random.normal(0.0, 1.0, self.n_max * self.Kc).reshape(self.n_max, self.Kc)
        G['cov'] = np.eye(self.Kc) #np.dot(self.G['mean'].T,self.G['mean']) 
        G['prodT'] = G['cov'] + self.n_max*G['cov'] #np.dot(self.G['mean'].T, self.G['mean']) + self.n_max*self.G['cov']
        if G_init is None: #If the matrix is not initialised
            self.G = G
        elif G_init['mean'].shape[0]<self.n_max: #If only the supervised part of the matrix is initialised
            self.G = G
            self.G['mean'][:G_init['mean'].shape[0],:] = G_init['mean']
            self.G['cov'] = G_init['cov']
            self.G['prodT'] = G_init['prodT']
        else: #If all the matrix is initialised          
            self.G = G_init
        
        # Initialization of the predictive space matrix Z
        Z['mean'] = np.random.normal(0.0, .01, self.n_max * self.Kp).reshape(self.n_max, self.Kp)
        Z['cov'] = np.eye(self.Kp) #np.dot(self.G['mean'].T,self.G['mean']) 
        Z['prodT'] = Z['cov'] + self.n_max*Z['cov'] #np.dot(self.G['mean'].T, self.G['mean']) + self.n_max*self.G['cov']
        if G_init is None: #If the matrix is not initialised
            self.Z = Z
        elif Z_init['mean'].shape[0]<self.n_max: #If only the supervised part of the matrix is initialised
            self.Z = Z
            self.Z['mean'][:Z_init['mean'].shape[0],:] = Z_init['mean']
            self.Z['cov'] = Z_init['cov']
            self.Z['prodT'] = Z_init['prodT']
        else: #If all the matrix is initialised          
            self.Z = Z_init
        
        for m in np.arange(self.m):
            # Initialization of the unknown data
            if self.SS[m]:
                self.tc[m] = np.random.rand(self.n_max, self.d[m])
                self.tS[m]['mean'] = np.random.randint(2, size=[self.n_max, self.d[m]])
                self.tS[m]['cov'] = np.eye(self.d[m]) 
                self.tS[m]['sumlogdet'] = 0
                self.XS[m]['mean'] = np.random.normal(0.0, 1.0, self.n_max * self.d[m]).reshape(self.n_max, self.d[m])
                # self.XS[m]['cov'] = np.eye(self.d[m]) 
                self.XS[m]['cov'] = np.zeros(self.XS[m]['mean'].shape)


        for m in np.arange(self.m):
            
            W[m]['mean'] = np.random.normal(np.zeros((self.d[m],self.Kc)), 1/(np.repeat(self.alpha_mean(m).reshape(1,self.Kc),self.d[m],axis=0))) #np.random.normal(0.0, 1.0, self.d[m] * self.Kc).reshape(self.d[m], self.Kc)
            W[m]['cov'] = np.dot(W[m]['mean'].T,W[m]['mean']) #np.eye(self.Kc)
            W[m]['prodT'] = np.dot(W[m]['mean'].T, W[m]['mean'])+self.Kc*W[m]['cov']
            if self.sparse[m]:
                W[m]['prodTalpha'] = np.zeros((self.d[m],))
                W[m]['prodTgamma'] = np.zeros((self.Kc,))
                W[m]['sumlogdet'] = 0
            
            # Initialization of the matrix V for each view
            V[m]['mean'] = np.random.normal(0.0, 0.1, self.Kp * self.d[m]).reshape(self.Kp, self.d[m])
            V[m]['cov'] = np.dot(V[m]['mean'].T,V[m]['mean']) #np.eye(self.Kc)
            V[m]['prodT'] = np.dot(V[m]['mean'].T, V[m]['mean'])+self.Kp*V[m]['cov']
            for k in np.arange(self.Kp):
                self.V_cov[m]['mean'][k] = np.eye(self.d[m])
            if self.sparse[m]:
                V[m]['prodTalpha'] = np.zeros((self.d[m],))
                V[m]['prodTgamma'] = np.zeros((self.Kp,))
                V[m]['sumlogdet'] = 0
            
            if method[m] == 'reg' or method[m] == 'mult':
                self.b[m]['cov'] = (1 + self.n[m] * self.tau_mean(m))**(-1) * np.eye(self.d[m])
                self.b[m]['mean'] = self.tau_mean(m) * np.dot(np.sum(self.X[m]['mean'] - np.dot(self.G['mean'], W[m]['mean'].T), axis=0)[np.newaxis,:], self.b[m]['cov'])
                self.b[m]['prodT'] = np.sum(self.b[m]['mean']**2) + self.d[m]*self.b[m]['cov'][0,0]    #mean of a noncentral chi-squared distribution
            else:
                self.b[m]['cov'] = np.zeros((self.d[m],self.d[m]))
                self.b[m]['mean'] = np.zeros((self.d[m],))
                self.b[m]['prodT'] = np.sum(self.b[m]['mean']**2) + self.d[m]*self.b[m]['cov'][0,0]   
                


        self.W = W if None in W_init else W_init
        self.V = V 
               
    def qGamma(self,a,b,m_i,r,mask=None,sp=[None]):
        """ Initialisation of variables with Gamma distribution..
    
        Parameters
        ----------
        __a : array (shape = [m_in, 1]).
            Initialistaion of the parameter a.        
        __b : array (shape = [m_in, 1]).
            Initialistaion of the parameter b.
        __m_i: int.
            Number of views. 
        __r: array (shape = [m_in, 1]).
            dimension of the parameter b for each view.
            
        """
        
        param = [None]*m_i
        for m in np.arange(m_i):
            if (None in sp) or sp[m] == 1:
                info = {                
                    "a":         a[m],
                    "LH":         None,
                    "ElogpWalp":  None,
                }
                if mask is None or mask[m] is None:
                    info["b"] = (b[m]*np.ones((r[m],))).flatten()
                else:
                    info["b"] = (b[m]*np.ones((len(np.unique(mask[m])),1))).flatten()
                param[m] = info
        return param

    def qGamma_uni(self,a,b,K):
        """ Initialisation of variables with Gamma distribution..
    
        Parameters
        ----------
        __a : array (shape = [1, 1]).
            Initialistaion of the parameter a.        
        __b : array (shape = [K, 1]).
            Initialistaion of the parameter b.
        __m_i: int.
            Number of views. 
        __K: array (shape = [K, 1]).
            dimension of the parameter b for each view.
            
        """
        
        param = {                
                "a":         a,
                "b":         b,
                "LH":         None,
                "ElogpWalp":  None,
            }
        return param
    
    def alpha_mean(self,m):
        """ Mean of alpha.
        It returns the mean value of the variable alpha for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        
        return self.alpha[m]["a"] / self.alpha[m]["b"]
    
    def tau_mean(self,m):
        """ Mean of tau.
        It returns the mean value of the variable tau for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        
        return self.tau[m]["a"] / self.tau[m]["b"]

    def gamma_mean(self,m):
        """ Mean of gamma.
        It returns the mean value of the variable gamma for the specified view.
    
        Parameters
        ----------
        __m : int.
            View that wants to be used.
            
        """
        
        return self.gamma[m]["a"] / self.gamma[m]["b"]
    
    def eta_mean(self):
        return self.eta['a'] / self.eta['b']
    def tauz_mean(self):
        return self.tauz['a'] / self.tauz['b']
    def psi_mean(self,m):
        return self.psi[m]["a"] / self.psi[m]["b"]
    def mu_mean(self,m):
        return self.mu[m]["a"] / self.mu[m]["b"]
    
    