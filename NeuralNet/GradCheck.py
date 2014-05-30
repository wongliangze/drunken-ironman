# -*- coding: utf-8 -*-
"""
Created on Fri May 30 11:08:25 2014

Checks gradients of functions defined in Cost and Transfer.
Differences should be on the order of 1e-10 or smaller.

For example:
linear : 5.50670620214e-14
tanh : 1.23840867239e-10
logistic : 9.36529679408e-11
softplus : 1.24865962776e-11
DK : 2.141990546e-12
DK_weighted : 2.4134077484e-12
MSE : 1.81749686946e-13
KL_tanh : 1.23581141234e-05
KL_logistic : 1.22888705843e-05

@author: Liang Ze
"""
from __future__ import division
import numpy as np
from numpy.linalg import norm

def computeNumericalGradient(J,theta): 
    ''' the method used here is the same as the one the Andrew Ng used 
        in his MATLAB code. Although theres a slight difference in the 
        implementation though. This code was koped from Daniel. 
    '''
    ''' theta is assumed to be a vector '''
        
    numgrad = np.zeros(np.size(theta))         
    EPSILON = 0.0001            
    
    for i in range(np.size(theta)):                 
        epsi = np.zeros(np.size(theta)) 
        epsi[i] = EPSILON/2
        epsi = epsi.reshape(np.shape(theta))
        numgrad[i] = (J(theta+epsi)-J(theta-epsi))/EPSILON                        
    
    return numgrad.reshape(np.shape(theta))

def compare(a,b):
    return norm(a-b)/norm(a+b) 

def compare_list(A,B):
    return [compare(a,b) for a,b in zip(A,B)]        
        

if __name__ == "__main__":
    import Transfer   
    import Cost
    import numpy as np
    
    # Transfer functions        
    for transfer in Transfer.transfer_list:
        x = np.random.rand(1)
        f = Transfer.assign(transfer)        
        numgrad = computeNumericalGradient(f,x)    
        grad = f.deriv(x)
        diff = compare(numgrad,grad)
        
        print "{0} : {1}".format(transfer,diff)
    
    # Param cost functions        
    w = np.random.rand(5,3)
    m = np.random.rand(*np.shape(w))
    param_func_args = [(w,),(w,m)]    
    
    for cost,args in zip(Cost.param_func_list,param_func_args):                                
        x = args[0]        
        f = Cost.assign(cost)
        other_args = args[1:]
        numgrad = computeNumericalGradient(lambda z: f(z,*other_args), x)
        grad = f.deriv_w(x,*other_args)
        diff = compare(numgrad,grad)
        
        print "{0} : {1}".format(cost,diff)
    
    # Layer cost functions
    data = np.random.rand(5,3)
    other = np.random.rand(*np.shape(data))
    mask = np.random.randint(0,2,np.shape(data))
    #mask = None
    sparse_rate = 0.1
    layer_func_args = [(data,other,mask),(data,sparse_rate,mask),(data,sparse_rate,mask)]    
    
    for cost,args in zip(Cost.layer_func_list,layer_func_args):                                
        x = args[0]        
        f = Cost.assign(cost)
        other_args = args[1:]
        numgrad = computeNumericalGradient(lambda z: f(z,*other_args), x)
        grad = f.delta(x,*other_args)
        diff = compare(numgrad,grad)
        
        print "{0} : {1}".format(cost,diff)
    
    
    
    
        
        