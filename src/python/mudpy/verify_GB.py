#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:40:57 2021

@author: kaslasm
"""
import numpy as np



def calc_distance_weight_matrix (source):
    '''
        calculate distance weight matrix  ||W_d||
        input coord = fault coordinates 
        '''

    from numpy import zeros
    import scipy.spatial.distance as sd
    from numpy import tile,sin,cos,deg2rad,sqrt

    R=6371.
    x=(R-source[:,3])*sin(deg2rad(90-source[:,2]))*cos(deg2rad(source[:,1]))
    y=(R-source[:,3])*sin(deg2rad(90-source[:,2]))*sin(deg2rad(source[:,1]))
    z=(R-source[:,3])*cos(deg2rad(90-source[:,2]))
    


    numfault = len(x)

    coord_mat = zeros((numfault*2,3))
    for ii in range (numfault):
        jj = ii*2 
        kk= ii*2 +1 
        coord_mat[jj,0] = x[ii]
        coord_mat[jj,1] = y[ii]
        coord_mat[jj,2] = z[ii]

        coord_mat[kk,0] = x[ii]
        coord_mat[kk,1] = y[ii]
        coord_mat[kk,2] = z[ii]

        
    dist_matr = sd.cdist(coord_mat, coord_mat, 'euclidean')
    return dist_matr 

def load_data (G_file, d_file):

    G = np.load(G_file)
    d = np.load(d_file) 

    return G,d

def apply_weights(weights, G, d): 

    size = len(d)
    W = np.eye(size) * weights
    WG = W.dot(G)
    wd = d * weights
    return WG, wd


def calculate_Gen_inv (WG):
    
    K=(WG.T).dot(WG)
    Kinv=K
    Gen_inv  = np.linalg.inv(Kinv).dot( WG.T) 
    return Gen_inv

def calculate_resolution_matrix(Gen_inv, G):
    R_matr = Gen_inv.dot (WG)
    return R_matr

def calculate_spread(D_matr,R_matr):
    spread = ( D_matr * R_matr**2 ).sum()
    return spread


# ---------------------------------------------------------------------------


    # Main program

fault_file= '/Users/kaslasm/Documents/projects/mudpy_testing/results/Nepal_example/data/model_info/nepal_10.fault'
G_file = '/Users/kaslasm/Documents/projects/mudpy_testing/results/Nepal_example/GFs/matrices/nepal.npy'
d_file = '/Users/kaslasm/Documents/projects/mudpy_testing/results/Nepal_example/GFs/matrices/data.npy'

source=np.loadtxt(fault_file,ndmin=2)

G, d = load_data(G_file, d_file)
WG, wd = apply_weights(0.5, G, d)
Gen_inv =calculate_Gen_inv(WG)

R_matr= calculate_resolution_matrix(Gen_inv, G)
D_matr = calc_distance_weight_matrix (source)
spread = calculate_spread(D_matr, R_matr)

print(spread)


