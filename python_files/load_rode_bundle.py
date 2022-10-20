#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:22:24 2022

@author: fiore
"""




def load_angeli ():
    
    
    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    
    fileName='/students/phd_ea/fiore/Diego/Diego/stats_forced.dat'
    
    array_txt = np.loadtxt(fileName)    

    nx=256
    ny=448


    x=np.zeros((nx,ny))
    y=np.zeros((nx,ny))
    U=np.zeros((nx,ny))
    V=np.zeros((nx,ny))
    W=np.zeros((nx,ny))
    T=np.zeros((nx,ny))
    rs_xx=np.zeros((nx,ny))
    rs_xy=np.zeros((nx,ny))
    rs_xz=np.zeros((nx,ny))
    rs_yy=np.zeros((nx,ny))
    rs_yz=np.zeros((nx,ny))
    rs_zz=np.zeros((nx,ny))
    thf_x=np.zeros((nx,ny))
    thf_y=np.zeros((nx,ny))
    thf_z=np.zeros((nx,ny))
    k_theta=np.zeros((nx,ny))
    dUdx=np.zeros((nx,ny))
    dUdy=np.zeros((nx,ny))
    dUdz=np.zeros((nx,ny))
    dVdx=np.zeros((nx,ny))
    dVdy=np.zeros((nx,ny))
    dVdz=np.zeros((nx,ny))
    dWdx=np.zeros((nx,ny))
    dWdy=np.zeros((nx,ny))
    dWdz=np.zeros((nx,ny))
    dTdx=np.zeros((nx,ny))
    dTdy=np.zeros((nx,ny))
    dTdz=np.zeros((nx,ny))
    uxux=np.zeros((nx,ny))
    uyuy=np.zeros((nx,ny))
    uzuz=np.zeros((nx,ny))
    vxvx=np.zeros((nx,ny))
    vyvy=np.zeros((nx,ny))
    vzvz=np.zeros((nx,ny))
    wxwx=np.zeros((nx,ny))
    wywy=np.zeros((nx,ny))
    wzwz=np.zeros((nx,ny))
    TxTx=np.zeros((nx,ny))
    TyTy=np.zeros((nx,ny))
    TzTz=np.zeros((nx,ny))



    count = 0

    for j in range(0,ny) :
        for i in range (0,nx):
            x[i,j]=array_txt[count,0]
            y[i,j]=array_txt[count,1]
            U[i,j]=array_txt[count,2]
            V[i,j]=array_txt[count,3]
            W[i,j]=array_txt[count,4]
            T[i,j]=array_txt[count,5]
            rs_xx[i,j]=array_txt[count,6]
            rs_xy[i,j]=array_txt[count,7]
            rs_xz[i,j]=array_txt[count,8]
            rs_yy[i,j]=array_txt[count,9]
            rs_yz[i,j]=array_txt[count,10]
            rs_zz[i,j]=array_txt[count,11]
            thf_x[i,j]=array_txt[count,12]
            thf_y[i,j]=array_txt[count,13]
            thf_z[i,j]=array_txt[count,14]   
            k_theta[i,j]=array_txt[count,15]
            dUdx[i,j]=array_txt[count,16]
            dUdy[i,j]=array_txt[count,17] 
            dUdz[i,j]=array_txt[count,18]
            dVdx[i,j]=array_txt[count,19]
            dVdy[i,j]=array_txt[count,20] 
            dVdz[i,j]=array_txt[count,21]
            dWdx[i,j]=array_txt[count,22]
            dWdy[i,j]=array_txt[count,23] 
            dWdz[i,j]=array_txt[count,24]
            dTdx[i,j]=array_txt[count,25]
            dTdy[i,j]=array_txt[count,26] 
            dTdz[i,j]=array_txt[count,27]
            uxux[i,j]=array_txt[count,28]
            uyuy[i,j]=array_txt[count,29] 
            uzuz[i,j]=array_txt[count,30]
            vxvx[i,j]=array_txt[count,31]
            vyvy[i,j]=array_txt[count,32] 
            vzvz[i,j]=array_txt[count,33]
            wxwx[i,j]=array_txt[count,34]
            wywy[i,j]=array_txt[count,35] 
            wzwz[i,j]=array_txt[count,36]
            TxTx[i,j]=array_txt[count,37]
            TyTy[i,j]=array_txt[count,38] 
            TzTz[i,j]=array_txt[count,39]
            count=count+1



    k = 0.5*(rs_xx + rs_yy + rs_zz)+1e-15
    epsilon = ((uxux+uyuy+uzuz+vxvx+vyvy+vzvz+wxwx+wywy+wzwz))/550 +1e-12
    k_theta = 0.5*k_theta + 1e-15
    epsilonT = (TxTx+TyTy+TzTz)/550/0.031+1e-12
    
    return x, y, U, V,W, T, rs_xx, rs_xy, rs_xz, rs_yy, rs_yz, rs_zz, thf_x, thf_y, thf_z, k, k_theta, dUdx, dUdy, dUdz, dVdx, dVdy, dVdz, dWdx, dWdy, dWdz, dTdx, dTdy, dTdz, epsilon, epsilonT


#%%

import numpy as np
import math

x, y, U, V, W, T, rs_xx, rs_xy, rs_xz, rs_yy, rs_yz, rs_zz, thf_x, thf_y, thf_z, k, k_theta, dUdx, dUdy, dUdz, dVdx, dVdy, dVdz, dWdx, dWdy, dWdz, dTdx, dTdy, dTdz, epsilon, epsilon_theta = load_angeli()


class Class_DNS (): pass


PoD = 1.4
phi = 2 * 3**0.5 / math.pi * PoD**2 - 1
R=1/2/phi
Ph = PoD*R

x_in=x.reshape(-1)
y_in=y.reshape(-1)
epsilon_theta_in=epsilon_theta.reshape(-1)
dTdx_in=dTdx.reshape(-1)
dTdy_in=dTdy.reshape(-1)


indices = np.where((x_in**2 + y_in**2 < (R)**2) | ((x_in-Ph)**2 + (y_in-math.sqrt(3)*Ph)**2 < (R)**2) | ((x_in+Ph)**2 + (y_in-math.sqrt(3)*Ph)**2 < (R)**2) | ((x_in-Ph)**2 + (y_in+math.sqrt(3)*Ph)**2 < (R)**2) | ((x_in+Ph)**2 + (y_in+math.sqrt(3)*Ph)**2 < (R)**2) )

index = np.where( x_in**2 + y_in**2 < (R+0.01)**2 )
x_in=np.delete(x_in, index)
y_in=np.delete(y_in, index)
epsilon_theta_in=np.delete(epsilon_theta_in, index)
dTdx_in=np.delete(dTdx_in, index)
dTdy_in=np.delete(dTdy_in, index)

index = np.where( (x_in-Ph)**2 + (y_in-math.sqrt(3)*Ph)**2 < (R+0.01)**2 )
x_in=np.delete(x_in, index)
y_in=np.delete(y_in, index)
epsilon_theta_in=np.delete(epsilon_theta_in, index)
dTdx_in=np.delete(dTdx_in, index)
dTdy_in=np.delete(dTdy_in, index)

index = np.where( (x_in+Ph)**2 + (y_in-math.sqrt(3)*Ph)**2 < (R+0.01)**2 )
x_in=np.delete(x_in, index)
y_in=np.delete(y_in, index)
epsilon_theta_in=np.delete(epsilon_theta_in, index)
dTdx_in=np.delete(dTdx_in, index)
dTdy_in=np.delete(dTdy_in, index)

index = np.where( (x_in-Ph)**2 + (y_in+math.sqrt(3)*Ph)**2 < (R+0.01)**2 )
x_in=np.delete(x_in, index)
y_in=np.delete(y_in, index)
epsilon_theta_in=np.delete(epsilon_theta_in, index)
dTdx_in=np.delete(dTdx_in, index)
dTdy_in=np.delete(dTdy_in, index)

index = np.where( (x_in+Ph)**2 + (y_in+math.sqrt(3)*Ph)**2 < (R+0.01)**2 )
x_in=np.delete(x_in, index)
y_in=np.delete(y_in, index)
epsilon_theta_in=np.delete(epsilon_theta_in, index)
dTdx_in=np.delete(dTdx_in, index)
dTdy_in=np.delete(dTdy_in, index)


x_p=x.reshape(-1)
y_p=y.reshape(-1)




points=np.concatenate((x_in.reshape(-1,1),y_in.reshape(-1,1)),axis=-1)
in_points=np.concatenate((x_p.reshape(-1,1),y_p.reshape(-1,1)),axis=-1)

from scipy.interpolate import griddata

epsilon_theta_new = griddata(points, epsilon_theta_in, in_points, method='linear')

epsilon_theta_new[indices]=np.nan

epsilon_theta_new=epsilon_theta_new.reshape(x.shape)

dTdx_p = griddata(points, dTdx_in, in_points, method='linear')
dTdy_p = griddata(points, dTdy_in, in_points, method='linear')

dTdx_p[indices]=0.0
dTdy_p[indices]=0.0
dTdx_p=dTdx_p.reshape(x.shape)
dTdy_p=dTdy_p.reshape(x.shape)




x, y, U, V, W, T, rs_xx, rs_xy, rs_xz, rs_yy, rs_yz, rs_zz, thf_x, thf_y, thf_z, k, k_theta, dUdx, dUdy, dUdz, dVdx, dVdy, dVdz, dWdx, dWdy, dWdz, dTdx, dTdy, dTdz, epsilon, epsilon_theta

Bundle_DNS = Class_DNS()
Bundle_DNS.x = x
Bundle_DNS.y = y
Bundle_DNS.U = U
Bundle_DNS.V = V
Bundle_DNS.W = W
Bundle_DNS.T = T
Bundle_DNS.rs_xx = rs_xx
Bundle_DNS.rs_xy = rs_xy
Bundle_DNS.rs_xz = rs_xz
Bundle_DNS.rs_yy = rs_yy
Bundle_DNS.rs_yz = rs_yz
Bundle_DNS.rs_zz = rs_zz
Bundle_DNS.thf_x = thf_x
Bundle_DNS.thf_y = thf_y
Bundle_DNS.thf_z = thf_z
Bundle_DNS.k = k
Bundle_DNS.kt = k_theta
Bundle_DNS.epsilon = epsilon
Bundle_DNS.epsilont = epsilon_theta