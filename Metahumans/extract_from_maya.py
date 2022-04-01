# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:53:43 2022

@author: Stevo

This script is used to extract the arrays from Maya Autodesk (it is run in Maya). 
The extracted numpy arrays correspond to neutral face , blendshape basis and corrective blendshapes of the first level.
"""

import os
import numpy as np
import pymel.core as pycore

bs_node = pycore.PyNode("head_lod0_mesh_blendShapes") 
neutral_mesh = pycore.PyNode("head_lod0_mesh") # mesh of the face - this might not be in a neutral position at the momment, so
# later we will have to set all the weights to 0 and obtain a real neutral face.
# Before running the next line, we need to select this node:
bs_node2 = pycore.PyNode("CTRL_expressions") 
names0 = [str(atr) for atr in pycore.listAttr(keyable=True)] # This should give over 200 names,
# so we need to check which ones produce zero offset (puppils) or are connected to the shoulder area.
N = 1505
m = len(names0)
n = 72147
path = r'C:\Users\User\Data'

## First we extract weights for each frame:
W = np.zeros((N,m)) # number of frames we have in this model
M = np.zeros((N,n)) # number of coordinates (i.e. 3*5386 the number of vertices) in the face
for i in range(N):
    pycore.currentTime(i)
    W[i]=[bs_node2.getAttr(nm) for nm in names0]
    verts = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()
    M[i] = verts
    
# Now we can also remove the controllers with zero offset:
W_offset = np.mean(W,axis=0)
W = W[:,W_offset>0]
names1 = []
for i in range(len(W_offset)):
    if W_offset[i] > 0:
        names1.append(names0[i])

# Now go back to the first frame and set all weights to zero, to extract the neutral face
pycore.currentTime(0)
for nm in names0:
    bs_node2.setAttr(nm,0)
neutral_verts = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()

# Extract each (delta) blendshape:
# The idea is to activate a single controller to 1 (or whatever is the max value) and all the rest to zero, and just take vertices
deltas = []
for nm in names1:
    bs_node2.setAttr(nm,1)
    shape_points = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()
    shape_delta = shape_points - neutral_verts
    deltas.append(shape_delta)
    bs_node2.setAttr(nm,0)
deltas = np.array(deltas)

# Now remove all those that do not affest the mesh (they are ususally connected only to pupils or teeth)
D_offset = np.sum(np.abs(deltas),axis=1)
deltas = deltas[D_offset>0]
names = []
for i in range(len(D_offset)):
    if D_offset[i] > 0:
        names.append(names1[i])
W = W[:,D_offset>0]

os.chdir(path)
np.save('weights.npy',W)
np.save('meshes.npy',M)
np.save('deltas.npy',deltas)
np.save('neutral.npy',neutral_verts)
    
### First level corrections:
for nm in names:
    bs_node2.setAttr(nm,0) 
m = len(names)
m2 = 642

first_level = {}
for i in range(m-1):
    bs_node2.setAttr(names[i],.5)
    for j in range(i+1,m):
        bs_node2.setAttr(names[j],.5)
        candidates = []
        c_values = []
        for c_shape in range(m2):
            if bs_node.w[c_shape].get() == .25:
                candidates.append(c_shape)
                c_values.append(bs_node.w[c_shape].get())
        if len(candidates) > 0:
            first_level[(i,j)] = [candidates, c_values]
        bs_node2.setAttr(names[j],0)
    bs_node2.setAttr(names[i],0)
    
fli = first_level.items()
keys = np.array([fli[i][0] for i in range(len(fli))])
corr_keys = np.array([fli[i][1][0] for i in range(len(fli))])

corr_shapes = []
for i in range(len(fli)):
    k1,k2 = keys[i][0],keys[i][1]
    bs_node2.setAttr(names[k1],1)    
    bs_node2.setAttr(names[k2],1)
    pred = neutral_verts + deltas[k1] + deltas[k2]
    true = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()
    corr_shapes.append(true-pred)
    bs_node2.setAttr(names[k1],0)    
    bs_node2.setAttr(names[k2],0)
    
corr_shapes = np.array(corr_shapes)
np.save('bs1.npy',corr_shapes)
np.save('keys1.npy',keys)
