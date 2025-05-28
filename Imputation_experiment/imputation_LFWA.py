import numpy as np
import os
import pickle
import argparse
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.cross_decomposition import CCA
from Model import OSIRIS
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
print(sklearn.__version__)
#Version = 1.0.2

from missingpy import MissForest


import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
(prv_fold,foldername) = os.path.split(dirpath)
os.sys.path.append(prv_fold +'/lib/')
os.sys.path.append(prv_fold +'\\lib\\')
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description = 'Trainer')
    parser.add_argument('--fold', dest = 'fold', help = 'Folds to test', default = 0, type = int)
    parser.add_argument('--perc', dest = 'perc', help = 'percentage of missing values', default = 5, type = int)
    parser.add_argument('--imputer', dest = 'imputer', help = 'imputation algorithm', default = 'None', type = str)
    parser.add_argument('--seed', dest = 'seed', help = 'random seed', default = 10, type = int)
    args = parser.parse_args()
    return args
args = parse_args()
print('Imputer: ', args.imputer, flush=True)
print('Fold: ', args.fold, flush=True)


import numpy as np
from scipy.io import savemat

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def impute_missing_values_VAE(train_matrix, test_matrix):

    input_dim = train_matrix.shape[1]
    

    train_mask = np.isnan(train_matrix)
    test_mask = np.isnan(test_matrix)
    

    scaler = StandardScaler()
    train_matrix_filled = np.nan_to_num(train_matrix)
    test_matrix_filled = np.nan_to_num(test_matrix)
    train_matrix_filled = scaler.fit_transform(train_matrix_filled)
    test_matrix_filled = scaler.transform(test_matrix_filled)

    inputs = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(inputs)
    encoded = layers.Dense(32, activation='relu')(encoded)
    z_mean = layers.Dense(16, name="z_mean")(encoded)
    z_log_var = layers.Dense(16, name="z_log_var")(encoded)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = layers.Lambda(sampling, output_shape=(16,), name="z")([z_mean, z_log_var])

    decoder_input = layers.Input(shape=(16,))
    decoded = layers.Dense(32, activation='relu')(decoder_input)
    decoded = layers.Dense(64, activation='relu')(decoded)
    outputs = layers.Dense(input_dim, activation='linear')(decoded) 
    
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    decoder = keras.Model(decoder_input, outputs, name="decoder")
    
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name="vae")
    vae.compile(optimizer='adam', loss='mse')
    

    vae.fit(train_matrix_filled, train_matrix_filled, epochs=200, batch_size=16, verbose=1)
    

    imputed_train = vae.predict(train_matrix_filled)
    imputed_test = vae.predict(test_matrix_filled)
    

    imputed_train = scaler.inverse_transform(imputed_train)
    imputed_test = scaler.inverse_transform(imputed_test)
    

    train_matrix[train_mask] = imputed_train[train_mask]
    test_matrix[test_mask] = imputed_test[test_mask]
    
    return train_matrix, test_matrix

################
#################

def add_random_nans(matrix, percentage, seed=42):
   
    np.random.seed(seed)
    total_elements = matrix.size
    num_nans = int(total_elements * (percentage / 100))
    

    indices = np.random.choice(total_elements, num_nans, replace=False)
    

    flat_matrix = matrix.flatten()
    flat_matrix = flat_matrix.astype(float) 
    flat_matrix[indices] = np.nan
    return flat_matrix.reshape(matrix.shape)




def calcAUC(Y_pred, Y_tst):
    n_classes = Y_pred.shape[1]
    
    # Compute ROC curve and ROC area for each class    
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros((n_classes,1))
    for i in np.arange(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_tst[:,i], Y_pred[:,i]/n_classes)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    return roc_auc.flatten()
    
def multAUC(Y_tst_bin, Y_pred):
    p_class = np.sum(Y_tst_bin,axis=0)/np.sum(Y_tst_bin)
    return np.sum(calcAUC(Y_pred, Y_tst_bin)*p_class)   

def linear_kernel(X1, X2):
		return np.dot(X1, X2.T)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


X1 = np.load(r'./Data/LFWA/X1.npy')
X2 = np.load(r'./Data/LFWA/X2.npy')
Y = np.load(r'./Data/LFWA/labels.npy')

X1 = add_random_nans(X1, args.perc, seed = 123)
X2 = add_random_nans(X2, args.perc, seed = 123)


database = 'LFWA'

# Create data partitions
folds = 10
# =================================================== #
# Don't run, just to generate folds and save in a file
# =================================================== #
folds_file = './Data/LFWA/'+str(folds)+'folds_'+database+'.p'
print('Loading test and validation folds.')
[fold_tst, dict_fold_val] = pickle.load(open(folds_file,'rb'))



pos_tr = fold_tst[args.fold][0]
pos_tst = fold_tst[args.fold][1]


X1_tr = X1[pos_tr,:]
X1_tst = X1[pos_tst,:]

X2_tr = X2[pos_tr,:]
X2_tst = X2[pos_tst,:]

Y_tr = Y[pos_tr]
Y_tst = Y[pos_tst]

Y_tr = Y_tr[:,np.newaxis]
Y_tst = Y_tst[:,np.newaxis]


if args.imputer == 'None': 
    print('No previous imputation')
elif args.imputer == 'median':
    print('Median imputation')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    X1_tr = imp_mean.fit_transform(X1_tr)
    X1_tst = imp_mean.transform(X1_tst)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    X2_tr = imp_mean.fit_transform(X2_tr)
    X2_tst = imp_mean.transform(X2_tst)

elif args.imputer == 'mean':
    print('Mean imputation')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X1_tr = imp_mean.fit_transform(X1_tr)
    X1_tst = imp_mean.transform(X1_tst)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X2_tr = imp_mean.fit_transform(X2_tr)
    X2_tst = imp_mean.transform(X2_tst)

elif args.imputer == 'MICE':
    print('MICE imputation')
    imp_mean = IterativeImputer(random_state=0)
    X1_tr = imp_mean.fit_transform(X1_tr)
    X1_tst = imp_mean.transform(X1_tst)
    imp_mean = IterativeImputer(random_state=0)
    X2_tr = imp_mean.fit_transform(X2_tr)
    X2_tst = imp_mean.transform(X2_tst)

    print('KNN imputation')
    imp_mean = KNNImputer(n_neighbors=2)
    X1_tr = imp_mean.fit_transform(X1_tr)
    X1_tst = imp_mean.transform(X1_tst)
    imp_mean = KNNImputer(n_neighbors=2)
    X2_tr = imp_mean.fit_transform(X2_tr)
    X2_tst = imp_mean.transform(X2_tst)

elif args.imputer == 'VAE':
    print('VAE imputation')
    X1_tr, X1_tst = impute_missing_values_VAE(X1_tr,X1_tst)
    X2_tr, X2_tst = impute_missing_values_VAE(X2_tr,X2_tst)

elif args.imputer == 'RF':
    print('RF imputation')
    imp_mean = MissForest(max_iter = 1)
    X1_tr = imp_mean.fit_transform(X1_tr)
    X1_tst = imp_mean.transform(X1_tst)
    imp_mean = MissForest(max_iter = 1)
    X2_tr = imp_mean.fit_transform(X2_tr)
    X2_tst = imp_mean.transform(X2_tst)

print('Imputation ready!', flush=True)

X1 = np.vstack((X1_tr, X1_tst))
X2 = np.vstack((X2_tr, X2_tst))

scaler1 = StandardScaler()
X1 = scaler1.fit_transform(X1)
scaler2 = StandardScaler()
X2 = scaler2.fit_transform(X2)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
train_encoded = encoder.fit_transform(Y_tr)

test_encoded = encoder.transform(Y_tst)

soft_predictions = np.zeros((np.shape(X1_tst)[0], np.shape(test_encoded)[1]))
elbos = []

print('Training...', flush=True)
for i in range(np.shape(test_encoded)[1]):
    myModel = OSIRIS.OSIRIS(Kc = 100, Kp = 1, prune = 1, SS_sep = 1, Yy = test_encoded[:,i,np.newaxis])
    X0_t = myModel.struct_data(X1, 'reg', 1)
    X1_t = myModel.struct_data(X2, 'reg', 1)
    Y1_tr = myModel.struct_data(train_encoded[:,i,np.newaxis], 'mult', 0)
    Y1_tst = myModel.struct_data(test_encoded[:,i,np.newaxis], 'mult', 0)

    myModel.fit(X0_t,X1_t, Y1_tr, max_iter = 20, Y_tst = Y1_tst, AUC = 0, ACC = 0, verbose = 1)

    Y_pred = myModel.compute_predictions(X_tst = [None], m_in=[0,1], m_out=2)
    Y_pred_tr = myModel.compute_predictions(X_tst = [None], m_in=[0,1], m_out=2, tr = 1)
    soft_predictions[:,i, np.newaxis] = myModel.X[-1]['mean'][-np.shape(X1_tst)[0]:]
    elbos.append(myModel.L[-1])

final_elbo = np.mean(elbos)

print('Final elbo: ',final_elbo)

hard_pred = np.argmax(soft_predictions, axis=1)

balanced_accuracy_tr = metrics.balanced_accuracy_score(Y_tst, hard_pred)
print('Balanced Accuracy: ', balanced_accuracy_tr)

results = [soft_predictions, Y_tst,final_elbo]
path_save = r'./Results/Imputation/LFWA/OSIRIS_'+str(args.seed)+'_'+str(args.fold)+'_'+str(args.perc)+'_'+str(args.imputer)+'.npy'
with open(path_save, "wb") as a:
    pickle.dump(results, a)




    