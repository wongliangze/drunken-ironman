from __future__ import division
import numpy as np
import operator
from numpy.linalg import norm
import time

# HELPER FUNCTIONS
def getfree(param,free):
    return list(param[np.ones_like(param)*free != 0.])
    
def setfree(param,free,new_param):
    param[np.ones_like(param)*free  != 0.] = np.ravel(new_param)
     

# NUMERICAL FUNCTIONS
def KL_logistic(x,sparse_rate):
    assert np.min(x*(1.-x) > 0), "Saturated KL computation"
    assert sparse_rate*(1. - sparse_rate) > 0, "sparse_rate must be strictly between 0 and 1"
    return sparse_rate*np.log(sparse_rate/x) + (1.-sparse_rate)*np.log((1.-sparse_rate)/(1.-x))

def dKL_logistic(x,sparse_rate):
    assert np.min(x*(1.-x) > 0), "Saturated dKL computation"
    assert sparse_rate*(1. - sparse_rate) > 0, "sparse_rate must be strictly between 0 and 1"
    return -sparse_rate/x+(1.-sparse_rate)/(1.-x)	

def KL_tanh(x,sparse_rate):
    assert np.min((1.+x)*(1.-x) > 0), "Saturated KL computation"
    assert (1.+sparse_rate)*(1. - sparse_rate) > 0, "sparse_rate must be strictly between 0 and 1"
    return 0.5*(1.-sparse_rate)*np.log((1.-sparse_rate)/(1.-x)) + 0.5*(1.+sparse_rate)*np.log((1.+sparse_rate)/(1.+x))


def dKL_tanh(x,sparse_rate):
    assert np.min((1.+x)*(1.-x) > 0), "Saturated KL computation"
    assert (1.+sparse_rate)*(1. - sparse_rate) > 0, "sparse_rate must be strictly between 0 and 1"
    return -0.5*(1.+sparse_rate)/(1.+x)+0.5*(1.-sparse_rate)/(1.-x)    
    
# RAMDOM INIT FUNCTIONS
def init_random_uniform(size_in,size_out,random_seed=None):
    random_state = np.random.RandomState(random_seed)
    r = np.sqrt(6)/np.sqrt(size_in+size_out+1)
    return random_state.rand(size_in,size_out)*2*r-r

    
# STRING PROCESSING FUNCTIONS
def None2Blank(inlist):
    return [('' if elm is None else elm) for elm in inlist]

def listify(item, num):
    if type(item) is not list:
        return [item]*num
    return item  

# LOADING AND SAVING
# Copied from Neurolab's tools.py
def load_net(fname):
    from pickle import load

    with open(fname, 'rb') as file:
        net = load(file)

    return net

def save_net(net, fname):
    from pickle import dump

    with open(fname, 'wb') as file:
        dump(net, file)

# Gradcheck FUNCTIONS
def computeNumericalGradient(J,theta): 
    ''' the method used here is the same as the one the Andrew Ng used 
        in his MATLAB code. Although theres a slight difference in the 
        implementation though. This code was koped from Daniel. 
    '''
    ''' theta is assumed to be a vector '''
        
    numgrad = np.zeros(np.size(theta))         
    EPSILON = 0.0001            
    
    t0 = time.time()
    epsi = np.zeros(np.size(theta)) 
    epsi[0] = EPSILON/2
    epsi = epsi.reshape(np.shape(theta))
    numgrad[0] = (J(theta+epsi)-J(theta-epsi))/EPSILON                        
    print 'Computing numgrad... {} dimensions... could take {} seconds'.format(np.size(theta),(time.time()-t0)*np.size(theta))
    
    for i in range(1,np.size(theta)):                 
        epsi = np.zeros(np.size(theta)) 
        epsi[i] = EPSILON/2
        epsi = epsi.reshape(np.shape(theta))
        numgrad[i] = (J(theta+epsi)-J(theta-epsi))/EPSILON                        
    
    return numgrad.reshape(np.shape(theta))

def compare(a,b):
    return norm(a-b)/norm(a+b) 

def compare_list(A,B):
    return [compare(a,b) for a,b in zip(A,B)]       