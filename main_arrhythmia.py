import numpy as np
import os
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.cross_decomposition import CCA
from Models.get_sn import get_sn 
from Models.model import CPMNets
from Models import classfiy 
from Models import OSIRIS
from Models.AverageMeter import AverageMeter
from scipy.io import savemat
from Models.utils import set_seed 
from Models.wrapper import train_CSMVIB 
from Models.config import get_args
from Models.dataload import ImportData
from sklearn.preprocessing import OneHotEncoder
from Models import import_data as impt
from Models.helper import f_get_minibatch_set, evaluate
from Models.class_DeepIMV_AISTATS import DeepIMV_AISTATS
from sklearn.model_selection import train_test_split
import tensorflow as tf

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Models.model_TMC import TMC
from Models.data import Multi_view_data
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from Models.model_CSMVIB import MVIB
from scipy.io import loadmat
from sklearn.preprocessing import label_binarize

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
(prv_fold,foldername) = os.path.split(dirpath)
os.sys.path.append(prv_fold +'/lib/')
os.sys.path.append(prv_fold +'\\lib\\')
from Models import fast_fs_ksshiba_b
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description = 'Trainer')
    parser.add_argument('--fold', dest = 'fold', help = 'Folds to test', default = 0, type = int)
    parser.add_argument('--seed', dest = 'seed', help = 'random seed', default = 500, type = int)
    parser.add_argument('--model', dest = 'model', help = 'model', default = 'CCA', type = str)
    args = parser.parse_args()
    return args
args = parse_args()
print('Model: ', args.model)
print('Fold: ', args.fold)
print('Seed: ', args.seed)

np.random.seed(args.seed)

import numpy as np
from scipy.io import savemat

def save_data_mat(X_train, X_test, Y_train, Y_test, name_file):
 

    Y_train = np.array(Y_train).reshape(-1, 1)
    Y_test = np.array(Y_test).reshape(-1, 1)

    datos = {
        'gt_train': Y_train,
        'gt_test': Y_test
    }

    for i, matriz in enumerate(X_train):
        datos[f'x{i+1}_train'] = np.array(matriz)
    
    for i, matriz in enumerate(X_test):
        datos[f'x{i+1}_test'] = np.array(matriz)

    savemat('./TMC_temporal/'+name_file, datos)


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

# In[]:
database = 'arrhythmia'


X = np.load(r'./Data/Arrhythmia/X_arrhythmia.npy')
Y = np.load(r'./Data/Arrhythmia/Y_arrhythmia.npy')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

idx = np.random.randint(0,2,Y.shape[0]).astype(int)
# Create data partitions
folds = 10
# =================================================== #
# Don't run, just to generate folds and save in a file
# =================================================== #
folds_file = './Data/Arrhythmia/'+str(folds)+'folds_'+database+'.p'
if os.path.exists(folds_file):
    print('Loading test and validation folds.')
    [fold_tst, dict_fold_val] = pickle.load(open(folds_file,'rb'))
else:
    print('Generating test and validation folds.\nCAREFUL, you need to always use the same partitions!')
    from sklearn.model_selection import StratifiedKFold
    skf_tst = StratifiedKFold(n_splits=folds, shuffle = True)
    # fold_tst =[f for  i, f in enumerate(skf_tst.split(X, Y))]
    fold_tst =[f for  i, f in enumerate(skf_tst.split(X, idx))]
    dict_fold_val = {}
    for ii, f_tst in enumerate(fold_tst):
        pos_tr = f_tst[0]
        skf_val = StratifiedKFold(n_splits=folds, shuffle = True)
        fold_val =[f for  i, f in enumerate(skf_val.split(X[pos_tr], idx[pos_tr]))]
        dict_fold_val[ii]=fold_val
    
    pickle.dump([fold_tst, dict_fold_val], open(folds_file, "wb" ))
# =================================================== #

pos_tr = fold_tst[args.fold][0]
pos_tst = fold_tst[args.fold][1]
# Definition of subsets
pos_all = np.arange(X.shape[1])
pos_sel1 = np.arange(15)
pos_sel2 = np.setdiff1d(pos_all,pos_sel1) 

X_tr = X[pos_tr,:]
X_tst = X[pos_tst,:]
Y_tr = Y[pos_tr,:]
Y_tst = Y[pos_tst,:]


X1_tr = X_tr[:,pos_sel1]
X1_tst = X_tst[:,pos_sel1]

X2_tr = X_tr[:,pos_sel2]
X2_tst = X_tst[:,pos_sel2]






if args.model == 'MVFA':
    print('Training...')
    X1 = np.vstack((X1_tr, X1_tst))
    X2 = np.vstack((X2_tr, X2_tst))

    print(np.shape(X1), np.shape(X2))

    scaler1 = StandardScaler()
    X1 = scaler1.fit_transform(X1)
    scaler2 = StandardScaler()
    X2 = scaler2.fit_transform(X2)

    myModel = fast_fs_ksshiba_b.SSHIBA(Kc = 200, prune = 1, SS_sep = 1)
    X0_t = myModel.struct_data(X1, 'reg', 1)
    X1_t = myModel.struct_data(X2, 'reg', 1)
    Y1_tr = myModel.struct_data(Y_tr, 'mult', 1)
    Y1_tst = myModel.struct_data(Y_tst, 'mult', 1)

    myModel.fit(X0_t,X1_t, Y1_tr, max_iter = 300, Y_tst = Y1_tst, AUC = 0, ACC = 0, verbose = 1)

    Y_pred = myModel.compute_predictions(X_tst = [None], m_in=[0,1], m_out=2)
    Y_pred_tr = myModel.compute_predictions(X_tst = [None], m_in=[0,1], m_out=2, tr = 1)
    auc_tst = metrics.roc_auc_score(Y_tst, Y_pred)

    fpr_tr, tpr_tr, thresholds_roc = metrics.roc_curve(Y_tr, Y_pred_tr)

    optimal_idx_tr = np.argmax(tpr_tr - fpr_tr)
    optimal_threshold_tr = thresholds_roc[optimal_idx_tr]

    print('Optimal threshold train: ', optimal_threshold_tr)

    predicted_classes_tr = (Y_pred >= optimal_threshold_tr).astype(int)
    

    balanced_accuracy_tr = metrics.balanced_accuracy_score(Y_tst, predicted_classes_tr)

    elbo = myModel.return_elbo()[-1]


    print(f"Balanced Accuracy: {balanced_accuracy_tr}")

    results = [Y_pred, Y_tst]
    path_save = r'./Results/Arrhythmia/MVFA_'+str(args.seed)+'_'+str(args.fold)+'.npy'
    with open(path_save, "wb") as a:
        pickle.dump(results, a)

elif args.model == 'MVPLS':
    scaler1 = StandardScaler()
    X1_tr = scaler1.fit_transform(X1_tr)
    X1_tst = scaler1.transform(X1_tst)
    scaler2 = StandardScaler()
    X2_tr = scaler2.fit_transform(X2_tr)
    X2_tst = scaler2.transform(X2_tst)
    X_tr = [X1_tr, X2_tr]
    X_tst = [X1_tst, X2_tst]

    from Models import MVPLS

    gamma = 1
    landa = 1
    alpha = 1

    print('Training final model...')

    myModel = MVPLS.LR_ARD()
    myModel.fit(X_tr, Y_tr, X_tst, Y_tst, alpha_in = alpha, gamma_in = gamma, landa_in = landa)
    print('Fold: ', args.fold)
    y_mean = myModel.y_mean
    X_tst_lab = [torch.tensor(xt) for xt in X_tst]
    label_pre, pred_score = myModel.predict(X_tst_lab,y_mean)
    Y_tst = Y_tst*2-1
    bal_acc = metrics.balanced_accuracy_score(Y_tst.ravel(),label_pre.ravel())
    roc = metrics.roc_auc_score(Y_tst, pred_score)

    print('Balanced accuracy: ', bal_acc)

    results = [label_pre, Y_tst]


    path_save = r'./Results/Arrhythmia/PLS_'+str(args.seed)+'_'+str(args.fold)+'.npy'
    with open(path_save, "wb") as a:
        pickle.dump(results, a)





elif args.model == 'MKL':
    print('Training...')

    scaler1 = StandardScaler()
    X1_tr = scaler1.fit_transform(X1_tr)
    X1_tst = scaler1.transform(X1_tst)
    scaler2 = StandardScaler()
    X2_tr = scaler2.fit_transform(X2_tr)
    X2_tst = scaler2.transform(X2_tst)
    X_tr = [X1_tr, X2_tr]
    X_tst = [X1_tst, X2_tst]

    from sklearn.metrics.pairwise import rbf_kernel
    combined_train_kernel = 0
    combined_test_kernel = 0
    # Train model
    for view_train, view_test in zip(X_tr, X_tst):
        K_train = rbf_kernel(view_train, view_train,gamma=0.1)
        K_test = rbf_kernel(view_test, view_train,gamma=0.1)
        combined_train_kernel += K_train
        combined_test_kernel += K_test
    X_train = np.hstack(X_tr)
    parameters = {'kernel':['precomputed'], 'C':[1e-3,1e-2,1e-1,1, 10,100,1000], 'probability':[True]}
    model = svm.SVC()
    clf = GridSearchCV(model, parameters)
    clf.fit(combined_train_kernel, Y_tr.reshape(-1))
    # Test model
    y_test = Y_tst.reshape(-1)
    y_pred = clf.predict(combined_test_kernel)
    y_pred_prob = clf.predict_proba(combined_test_kernel)

    print('Balanced acuracy: ',metrics.balanced_accuracy_score(y_test, y_pred))

    results = [y_pred_prob, Y_tst]
    path_save = r'./Results/Arrhythmia/MKL_'+str(args.seed)+'_'+str(args.fold)+'.npy'
    with open(path_save, "wb") as a:
        pickle.dump(results, a)






elif args.model == 'CPM':
    print('Training...')
    scaler1 = StandardScaler()
    X1_tr = scaler1.fit_transform(X1_tr)
    X1_tst = scaler1.transform(X1_tst)
    scaler2 = StandardScaler()
    X2_tr = scaler2.fit_transform(X2_tr)
    X2_tst = scaler2.transform(X2_tst)
    X_tr = [X1_tr, X2_tr]
    X_tst = [X1_tst, X2_tst]


    X_train= {'data': {}}
    X_test = {'data': {}}

    views = ['0', '1']
    view_num = 2
    num = 0
    for name_of_view in views:
        X_train['data'][name_of_view] = X_tr[num]
        # view-specific test data
        X_test['data'][name_of_view]= X_tst[num]
        num += 1
    
    print(X_train)

    num_examples_tr = np.shape(X_train['data']['1'])[0] 
    num_examples_tst = np.shape(X_test['data']['1'])[0] 

    missing_rate = 0
    epochs_train = 1000
    epochs_test = 500
    lsd_dim = 128
    lamb = 1

    print('Start training')
    outdim_size = [X_train['data'][str(i)].shape[1] for i in range(view_num)]
    # set layer size
    layer_size = [[300, outdim_size[i]] for i in range(view_num)]
    # set parameter
    epoch = [epochs_train, epochs_test]
    learning_rate = [0.01, 0.001]
    # Randomly generated missing matrix
    Sn = get_sn(view_num, num_examples_tr + num_examples_tst, missing_rate)
    Sn_train = Sn[np.arange(num_examples_tr)]
    Sn_test = Sn[np.arange(num_examples_tst) + num_examples_tr]
    # Model building
    model = CPMNets(view_num, num_examples_tr, num_examples_tst, layer_size, lsd_dim, learning_rate,
                    lamb)
    # train
    model.train(X_train['data'], Sn_train, Y_tr.reshape(num_examples_tr), epoch[0])
    H_train = model.get_h_train()
    # test
    model.test(X_test['data'], Sn_test, Y_tst.reshape(num_examples_tst), epoch[1])
    H_test = model.get_h_test()
    label_pre, prob = classfiy.ave(H_train, H_test, Y_tr)
    

    results = [prob, Y_tst]
    print('Balanced accuracy: ', metrics.balanced_accuracy_score(Y_tst,label_pre))
    path_save = r'./Results/Arrhythmia/CPM_'+str(args.seed)+'_'+str(args.fold)+'.npy'
    with open(path_save, "wb") as a:
        pickle.dump(results, a)


elif args.model == 'CSMVIB':
    print('Training...')
    
    scaler1 = StandardScaler()
    X1_tr = scaler1.fit_transform(X1_tr)
    X1_tst = scaler1.transform(X1_tst)
    scaler2 = StandardScaler()
    X2_tr = scaler2.fit_transform(X2_tr)
    X2_tst = scaler2.transform(X2_tst)

    X1 = np.concatenate((X1_tr, X1_tst), 0)
    X2 = np.concatenate((X2_tr, X2_tst), 0)
    X_views = [X1, X2]


    data = {
        'X': [[X for X in X_views]], 
        'Y': Y
    }


    data['X'][0] = np.empty((len(X_views),), dtype=np.object)
    for i, X in enumerate(X_views):
        data['X'][0][i] = X


    data_name = 'datos_CSMVIB_'+str(args.fold)+'_'+database+'.mat'
    output_file = './CSMVIB_temporal/' + data_name
    savemat(output_file, data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dataset_folder = './CSMVIB_temporal'
    file_list = os.listdir(dataset_folder)

    file_path = os.path.join(dataset_folder, data_name)
    data = loadmat(file_path)

    labels = torch.tensor(np.squeeze(data['Y']) - 1).long() if np.min(data['Y']) != 0 else torch.tensor(np.squeeze(data['Y'])).long()
    data_set = ImportData(data_path=file_path)
    set_seed(123)

    input_dims = [data['X'][0][i].shape[1] for i in range(data['X'][0].shape[0])]
    class_num = len(np.unique(labels))
    sample_number = data['X'][0][0].shape[0]
    view_number = len(input_dims)


    train_idxs = range(np.shape(X1_tr)[0])
    test_idxs = np.arange(np.shape(X1_tr)[0], np.shape(X1)[0])

    net = MVIB(input_dims=input_dims, class_num=class_num).to(device)
    train_subset = torch.utils.data.Subset(data_set, train_idxs)
    test_subset = torch.utils.data.Subset(data_set, test_idxs)
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=50, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=50, shuffle=False, num_workers=0)
    report, y_pred, y_true, soft = train_CSMVIB(trainloader, testloader, 0, net, class_num)

    print('Balanced accuracy: ',metrics.balanced_accuracy_score(Y_tst, y_pred))

    results = [soft, Y_tst]
    path_save = r'./Results/Arrhythmia/CSMVIB_'+str(args.seed)+'_'+str(args.fold)+'.npy'
    with open(path_save, "wb") as a:
        pickle.dump(results, a)




elif args.model == 'TMC':
    print('Training...')

    scaler1 = StandardScaler()
    X1_tr = scaler1.fit_transform(X1_tr)
    X1_tst = scaler1.transform(X1_tst)
    scaler2 = StandardScaler()
    X2_tr = scaler2.fit_transform(X2_tr)
    X2_tst = scaler2.transform(X2_tst)
    X_tr = [X1_tr, X2_tr]
    X_tst = [X1_tst, X2_tst]

    save_data_mat(X_tr, X_tst, Y_tr, Y_tst, 'datos_TMC_'+str(args.fold)+'_'+database+'.mat')

    data_name = 'datos_TMC_'+str(args.fold)+'_'+database
    data_path = './TMC_temporal/' + data_name
    dims = [[np.shape(X1_tr)[1]], [np.shape(X2_tr)[1]]]
    views = len(dims)

    train_loader = torch.utils.data.DataLoader(
        Multi_view_data(data_path, train=True), batch_size=50, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        Multi_view_data(data_path, train=False), batch_size=50, shuffle=False)
    N_mini_batches = len(train_loader)
    print('The number of training images = %d' % N_mini_batches)

    model = TMC(len(np.unique(Y_tr)), views, dims, 50)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)


    # model.cuda()

    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num])
            target = Variable(target.long())
            # refresh the optimizer
            optimizer.zero_grad()
            evidences, evidence_a, loss = model(data, target, epoch)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())


    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        softs = []
        for batch_idx, (data, target) in enumerate(test_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num])
            data_num += target.size(0)
            with torch.no_grad():
                target = Variable(target.long())
                evidences, evidence_a, loss = model(data, target, epoch)
                _, predicted = torch.max(evidence_a.data, 1)
                softs.append(evidence_a.data)
                correct_num += (predicted == target).sum().item()
                balanced = metrics.balanced_accuracy_score(predicted,target)
                loss_meter.update(loss.item())

        print('====> acc: {:.4f}'.format(correct_num/data_num))
        return loss_meter.avg, correct_num/data_num, balanced, softs

    for epoch in range(1, 1000 + 1):
        train(epoch)

    test_loss, acc, bal_acc, soft = test(epoch)
    print('====> acc: {:.4f}'.format(bal_acc))
    soft = np.concatenate(soft)
    print('Balanced accuracy: ', bal_acc)

    results = [soft, Y_tst]
    path_save = r'./Results/XRay/TMC_'+str(args.fold)+'_'+str(args.seed)+'.npy'
    with open(path_save, "wb") as a:
        pickle.dump(results, a)


elif args.model == 'DeepIMV':
    print('Training...')


    scaler1 = StandardScaler()
    X1_tr = scaler1.fit_transform(X1_tr)
    X1_tst = scaler1.transform(X1_tst)
    scaler2 = StandardScaler()
    X2_tr = scaler2.fit_transform(X2_tr)
    X2_tst = scaler2.transform(X2_tst)
    X_tr = [X1_tr, X2_tr]
    X_tst = [X1_tst, X2_tst]


    va_X_set = {}
    tr_X_set = {}
    te_X_set = {}

    tr_Y_onehot = Y_tr
    
    te_Y_onehot = Y_tst

    M  = len(X_tr)
    tr_M = np.ones([np.shape(X_tr[0])[0], M])

    M  = len(X_tst)
    te_M = np.ones([np.shape(X_tst[0])[0], M])


    for m in range(len(X_tr)):
        tr_X_set[m] = X_tr[m]
        te_X_set[m] = X_tst[m]
        tr_X_set[m],va_X_set[m] = train_test_split(tr_X_set[m], test_size=0.2, random_state=42)
    
    tr_Y_onehot,va_Y_onehot, tr_M,va_M = train_test_split(tr_Y_onehot, tr_M, test_size=0.2, random_state=42)

    #CONTINUAR POR AQUI
    x_dim_set    = [tr_X_set[m].shape[1] for m in range(len(tr_X_set))]
    y_dim        = np.shape(tr_Y_onehot)[1]

    if y_dim == 1:
        y_type       = 'continuous'
    elif y_dim == 2:
        y_type       = 'binary'
    else:
        y_type       = 'categorical'
    
    
    mb_size         = 32
    steps_per_batch = int(np.shape(tr_M)[0]/mb_size) #for moving average
    
    input_dims = {
        'x_dim_set': x_dim_set,
        'y_dim': y_dim,
        'y_type': y_type,
        'z_dim': 50,

        'steps_per_batch': steps_per_batch
    }

    network_settings = {
        'h_dim_p1': 100,
        'num_layers_p1': 2,   #view-specific

        'h_dim_p2': 100,
        'num_layers_p2': 2,  #multi-view

        'h_dim_e': 100,
        'num_layers_e': 3,

        'fc_activate_fn': tf.nn.relu,
        'reg_scale': 0.,
    }
    

    lr_rate         = 1e-4
    iteration       = 5000
    stepsize        = 100
    max_flag        = 20

    k_prob          = 0.7
    
    alpha           = 1.0
    beta            = 0.01
    
    # save_path       = args.save_path
    
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)


    tf.reset_default_graph()
    gpu_options = tf.GPUOptions()
    
    sess  = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    model = DeepIMV_AISTATS(sess, "DeepIMV_AISTATS", input_dims, network_settings)
    

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    ##### TRAINING
    min_loss  = 1e+8   
    max_acc   = 0.0

    tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 0, 0, 0, 0, 0, 0
    va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc = 0, 0, 0, 0, 0, 0
    
    stop_flag = 0
    for itr in range(iteration):
        x_mb_set, y_mb, m_mb          = f_get_minibatch_set(mb_size, tr_X_set, tr_Y_onehot, tr_M)     

        _, Lt, Lp, Lkl, Lps, Lkls, Lc = model.train(x_mb_set, y_mb, m_mb, alpha, beta, lr_rate, k_prob)

        tr_avg_Lt   += Lt/stepsize
        tr_avg_Lp   += Lp/stepsize
        tr_avg_Lkl  += Lkl/stepsize
        tr_avg_Lps  += Lps/stepsize
        tr_avg_Lkls += Lkls/stepsize
        tr_avg_Lc   += Lc/stepsize


        x_mb_set, y_mb, m_mb          = f_get_minibatch_set(min(np.shape(va_M)[0], mb_size), va_X_set, va_Y_onehot, va_M)       
        Lt, Lp, Lkl, Lps, Lkls, Lc, _, _    = model.get_loss(x_mb_set, y_mb, m_mb, alpha, beta)

        va_avg_Lt   += Lt/stepsize
        va_avg_Lp   += Lp/stepsize
        va_avg_Lkl  += Lkl/stepsize
        va_avg_Lps  += Lps/stepsize
        va_avg_Lkls += Lkls/stepsize
        va_avg_Lc   += Lc/stepsize

        if (itr+1)%stepsize == 0:
            y_pred, y_preds = model.predict_ys(va_X_set, va_M)

    #         score = 

            print( "{:05d}: TRAIN| Lt={:.3f} Lp={:.3f} Lkl={:.3f} Lps={:.3f} Lkls={:.3f} Lc={:.3f} | VALID| Lt={:.3f} Lp={:.3f} Lkl={:.3f} Lps={:.3f} Lkls={:.3f} Lc={:.3f} score={}".format(
                itr+1, tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc,  
                va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc, evaluate(va_Y_onehot, np.mean(y_preds, axis=0), y_type))
                 )

            if min_loss > va_avg_Lt:
                min_loss  = va_avg_Lt
                stop_flag = 0
                # saver.save(sess, save_path  + 'best_model')
                print('saved...')
            else:
                stop_flag += 1

            tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 0, 0, 0, 0, 0, 0
            va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc = 0, 0, 0, 0, 0, 0

            if stop_flag >= max_flag:
                break

    print('FINISHED...')
    
    
    ##### TESTING
    # saver.restore(sess, save_path  + 'best_model')
    
    _, pred_ys = model.predict_ys(te_X_set, te_M)
    pred_y = np.mean(pred_ys, axis=0)
    print(np.shape(pred_y))
    print(np.shape(Y_tst))

    print('Test Score: {}'.format(evaluate(te_Y_onehot, pred_y, y_type)))

    results = [pred_y, Y_tst]
    path_save = r'./Results/Arrhythmia/DeepIMV_'+str(args.seed)+'_'+str(args.fold)+'.npy'
    with open(path_save, "wb") as a:
        pickle.dump(results, a)



elif args.model == 'OSIRIS':
    print('Training...')


    X1 = np.vstack((X1_tr, X1_tst))
    X2 = np.vstack((X2_tr, X2_tst))

    scaler1 = StandardScaler()
    X1 = scaler1.fit_transform(X1)
    scaler2 = StandardScaler()
    X2 = scaler2.fit_transform(X2)

    myModel = OSIRIS.OSIRIS(Kc = 200, Kp = 1, prune = 1, SS_sep = 1, Yy = Y_tst)
    X0_t = myModel.struct_data(X1, 'reg', 1)
    X1_t = myModel.struct_data(X2, 'reg', 1)
    Y1_tr = myModel.struct_data(Y_tr, 'mult', 1)
    Y1_tst = myModel.struct_data(Y_tst, 'mult', 1)

    myModel.fit(X0_t,X1_t, Y1_tr, max_iter = 500, Y_tst = Y1_tst, AUC = 0, ACC = 0, verbose = 1)

    Y_pred = myModel.compute_predictions(X_tst = [None], m_in=[0,1], m_out=2)
    Y_pred_tr = myModel.compute_predictions(X_tst = [None], m_in=[0,1], m_out=2, tr = 1)
    auc_tst = metrics.roc_auc_score(Y_tst, Y_pred)

    fpr_tr, tpr_tr, thresholds_roc = metrics.roc_curve(Y_tr, Y_pred_tr)

    optimal_idx_tr = np.argmax(tpr_tr - fpr_tr)
    optimal_threshold_tr = thresholds_roc[optimal_idx_tr]


    predicted_classes_tr = (Y_pred >= optimal_threshold_tr).astype(int)


    balanced_accuracy_tr = metrics.balanced_accuracy_score(Y_tst, predicted_classes_tr)

    elbo = myModel.return_elbo()[-1]


    print(f"Balanced Accuracy: {balanced_accuracy_tr}")


    q = myModel.q_dist

    elbo = myModel.L[-1]


    
    results = [Y_pred, Y_tst, elbo]
    path_save = r'./Results/Arrhythmia/OSIRIS_'+str(args.seed)+'_'+str(args.fold)+'.npy'
    with open(path_save, "wb") as a:
        pickle.dump(results, a)




    