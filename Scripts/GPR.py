# -*- coding: utf-8 -*-
"""
@author: Stevo
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct

def vtx_to_coord_idx(vtx_idx):
    coord_idx = []
    for i in range(len(vtx_idx)):
        vl = vtx_idx[i]
        coord_idx.append(3*vl)
        coord_idx.append(3*vl+1)
        coord_idx.append(3*vl+2)
    return np.array(coord_idx)

def GPR_pred(Xtr,ytr,Xtst):
    pred = GaussianProcessRegressor(kernel=DotProduct()).fit(Xtr, ytr).predict(Xtst)
    pred[pred<0] = 0
    pred[pred>1] = 1
    return pred

def GPR_clusters(clust_dict,X_train,X_test,y_train):
    N2 = len(X_test)
    m = y_train.shape[1]
    num_clusters = len(clust_dict.keys())
    final_y,final_count = np.zeros((N2,m)),np.zeros(m)
    for cluster in range(num_clusters):
        coord_idx = vtx_to_coord_idx(clust_dict[cluster][1])
        ctr_idx = clust_dict[cluster][0]
        if len(ctr_idx)>0:
            Xtr,Xtst = X_train[:,coord_idx],X_test[:,coord_idx]
            ytr = y_train[:,ctr_idx]
            ypred = GPR_pred(Xtr,ytr,Xtst)
            final_y[:,ctr_idx] += ypred
            final_count[ctr_idx] += 1
    final_y /= final_count
    return final_y
