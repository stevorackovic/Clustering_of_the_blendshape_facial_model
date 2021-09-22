# -*- coding: utf-8 -*-
"""
@author: Stevo
"""

import os
import numpy as np
import pymel.core as pycore
model_name = 'Head_Male'
results_path = r'~\Results'
os.chdir(results_path)
bs_node      = pycore.PyNode("head_mesh_blendShapes")
bs_node2     = pycore.PyNode("CTRL_expressions")
neutral_mesh = pycore.PyNode("head_mesh")
names        = [] # paste controller names here
  
len_test, n, m = 129, 16158, 147
cluster_num_choice = [2,5,10,13,17,20,25,30,40,50,75,100]
cluster_num_choice2 = [2,5,10,13,17,17,20,19,24,20,22,20]

predictions = np.load('y_pred_'+model_name+'_1.npy')
X_pred = np.zeros((len_test,n))
for i in range(len_test):
    pycore.currentTime(i)
    pp = predictions[i]
    for j in range(m):
        nm = names[j]
        bs_node2.setAttr(nm,pp[j])
    verts = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()
    X_pred[i] += verts
np.save('X_pred_'+model_name+'_1.npy',X_pred)

for i in range(len(cluster_num_choice)):
    num_cluster = cluster_num_choice[i]
    num_cluster2 = cluster_num_choice2[i]
    predictions = np.load('y_pred_'+model_name+'_'+str(num_cluster)+'_into_'+str(num_cluster2)+'.npy')
    X_pred = np.zeros((len_test,n))
    for i in range(len_test):
        pycore.currentTime(i) 
        pp = predictions[i]
        for j in range(m):
            nm = names[j]
            bs_node2.setAttr(nm,pp[j])
        verts = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()
        X_pred[i] += verts
    np.save('X_pred_'+model_name+'_'+str(num_cluster)+'_into_'+str(num_cluster2)+'.npy',X_pred)

