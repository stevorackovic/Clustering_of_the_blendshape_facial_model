# -*- coding: utf-8 -*-
"""
@author: Stevo
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from Clustering import complete_clustering
from GPR import GPR_pred, GPR_clusters
import warnings

# Ucitam podatke:
# neutral:  Neutral face, np.array(n)
# deltas:   Delta blendshapes, np.array(n,m) --- contains delta offsets.
# meshes:   Animation meshes, np.array(N,n)
# weights:  Controllers' weights over the animation, np.array(N,m)
os.chdir(r'~\Data')
neutral    = np.load('neutral.npy')
deltas     = np.load('deltas.npy')
n,m        = deltas.shape
weights    = np.load('weights.npy')
N          = len(weights)
meshes     = np.load('meshes.npy')
test,train = np.load('train.npy'),np.load('test.npy')

model_name = 'Head_Male'
results_path = r'~\Results'
clust_num_choice = [2,5,10,13,17,20,25,30,40,50,75,100]

X_train = meshes[train]
y_train = weights[train]
X_test  = meshes[test]
y_test  = weights[test]

def Evaluation_1(results_path,model_name,X_train,X_test,y_train,y_test,deltas,neutral,clust_num_choice):

    os.chdir(results_path)
    Ctr_pred_errors = []
    Pred_times      = []
    Ctr_per_clst    = []
    Vtx_per_clst    = []
    Vtx_total       = []
    
    N1,n = X_train.shape
    N1,m = y_train.shape
    Ctr_per_clst.append(m)
    Vtx_per_clst.append(n/3)
    Vtx_total.append(n/3)
    y_pred, pred_time = GPR_pred(X_train,y_train,X_test)
    Pred_times.append(pred_time)
    np.save('y_pred_'+model_name+'_1.npy', y_pred)
    ctr_error = np.linalg.norm(y_pred-y_test,axis=1).mean()
    Ctr_pred_errors.append(ctr_error)
    
    for i in range(len(clust_num_choice)):
        num_clusters = clust_num_choice[i]
        print('Working on num_clusters = ',num_clusters)
        tol_factor = .75
        clust_dict = complete_clustering(deltas,0,0,num_clusters,neutral,m,factor=tol_factor,merge=True)
        num_clusters2 = len(clust_dict.keys())
        
        Ctr_per_clst.append(np.mean([len(clust_dict[cluster][0]) for cluster in range(num_clusters2)]))
        Vtx_per_clst.append(np.mean([len(clust_dict[cluster][1]) for cluster in range(num_clusters2)]))
        Vtx_total.append(np.sum([len(clust_dict[cluster][1]) for cluster in range(num_clusters2) if len(clust_dict[cluster][0])>0]))
        
        y_pred, pred_time = GPR_clusters(clust_dict,X_train,X_test,y_train)
        Pred_times.append(pred_time)
        np.save('y_pred_'+model_name+'_'+str(num_clusters)+'_into_'+str(num_clusters2)+'.npy', y_pred)
        ctr_error = np.linalg.norm(y_pred-y_test,axis=1).mean()
        Ctr_pred_errors.append(ctr_error)
        
        for i in range(num_clusters2):
            np.save('Num_'+str(num_clusters)+'_into_'+str(num_clusters2)+'_clust_'+str(i)+'_controllers.npy',np.array(clust_dict[i][0]))
            np.save('Num_'+str(num_clusters)+'_into_'+str(num_clusters2)+'_clust_'+str(i)+'_vertices.npy',clust_dict[i][1])
          
    Results_matrix = np.array([Ctr_pred_errors,Pred_times,Ctr_per_clst,Vtx_per_clst,Vtx_total])
    np.save(model_name+'_evaluation_1.npy',Results_matrix)
    return Results_matrix

Results1 = Evaluation_1(results_path,model_name,X_train,X_test,y_train,y_test,deltas,neutral,clust_num_choice)
 
plt.figure(figsize=(6,3))
plt.plot(Results1[0])
plt.scatter(np.arange(13),Results1[0])
plt.xticks(np.arange(13),[1]+clust_num_choice)
plt.title('Controller Error over Number of Clusters')
plt.show()

plt.figure(figsize=(6,3))
plt.plot(Results1[1])
plt.scatter(np.arange(13),Results1[1])
plt.xticks(np.arange(13),[1]+clust_num_choice)
plt.title('Prediction Time over Number of Clusters')
plt.show()

plt.figure(figsize=(6,3))
plt.plot(Results1[2])
plt.scatter(np.arange(13),Results1[2])
plt.xticks(np.arange(13),[1]+clust_num_choice)
plt.title('Number of Controllers per Cluster')
plt.show()

plt.figure(figsize=(6,3))
plt.plot(Results1[3])
plt.scatter(np.arange(13),Results1[3])
plt.xticks(np.arange(13),[1]+clust_num_choice)
plt.title('Number of Vertices per Cluster')
plt.show()

plt.figure(figsize=(6,3))
plt.plot(Results1[4])
plt.scatter(np.arange(13),Results1[4])
plt.xticks(np.arange(13),[1]+clust_num_choice)
plt.title('Number of Considered Vertices over Number of Clusters')
plt.show()

def Evaluation_2(results_path,model_name,X_test,meshes,clust_num_choice,clust_num_choice2,neutral):
    os.chdir(results_path)
    Rec_errors  = [] 
    Mesh_errors = [] 
    X_pred      = np.load('X_pred_'+str(model_name)+'_1.npy') - neutral
    Mesh_errors.append(np.linalg.norm(X_pred-X_test,axis=1).mean())
    Rec_errors.append(0)
    
    for i in range(len(clust_num_choice)):
        num_clusters = clust_num_choice[i]
        num_clusters2 = clust_num_choice2[i]
        print('Working on num_clusters = ',num_clusters)
        X_pred = np.load('X_pred_'+str(model_name)+'_'+str(num_clusters)+'_into_'+str(num_clusters2)+'.npy') - neutral
        Mesh_errors.append(np.linalg.norm(X_pred-X_test,axis=1).mean())
    Results_matrix = np.array(Mesh_errors)
    np.save(model_name+'_evaluation_2.npy',Results_matrix)
    return Results_matrix
        
# clust_num_choice2 = [2,5,10,13,17,17,20,19,24,20,22,20]
# Results2 = Evaluation_2(results_path,model_name,X_test,meshes,clust_num_choice,clust_num_choice2,neutral)
# plt.figure(figsize=(6,3))
# plt.plot(Results2)
# plt.scatter(np.arange(13),Results2)
# plt.xticks(np.arange(13),[1]+clust_num_choice)
# plt.title('mesh')
# plt.show()
