# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:32:18 2014

@author: Liang Ze
"""

from __future__ import division
import numpy as np
import Cost
import Estimator
from test_grad import compare_list

def init_random_uniform(hidden_size,visible_size,random_seed=None):
    random_state = np.random.RandomState(random_seed)
    r = np.sqrt(6)/np.sqrt(hidden_size+visible_size+1)
    W1 = random_state.rand(visible_size,hidden_size)*2*r-r
    W2 = random_state.rand(hidden_size,visible_size)*2*r-r
    b1 = np.zeros((hidden_size))
    b2 = np.zeros((visible_size))
    return W1,W2,b1,b2 

def forward_pass_sig(data,W,b):        
    return sigmoid(np.dot(data,W) + b)
def forward_pass_conditional(data,W,b,D):
    return sigmoid(np.dot(data.filled(0),W) + np.dot(~data.mask,D) + b)  
def forward_pass_tanh(data,W,b):        
    return tanh(np.dot(data,W) + b)

def sigmoid(x):
    return 1/(1+np.exp(-x))    
def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))    

def KL_tanh(x,sparse_rate):
    return 0.5*(1-sparse_rate)*np.log((1-sparse_rate)/(1-x)) + 0.5*(1+sparse_rate)*np.log((1+sparse_rate)/(1+x))
def dKL_tanh(x,sparse_rate):
    return -0.5*(1+sparse_rate)/(1+x)+0.5*(1-sparse_rate)/(1-x)
def KL_logistic(x,sparse_rate):
    return sparse_rate*np.log(sparse_rate/x) + (1-sparse_rate)*np.log((1-sparse_rate)/(1-x))
def dKL_logistic(x,sparse_rate):
    return -sparse_rate/x+(1-sparse_rate)/(1-x)

# Cost functions
DK = Cost.assign("DK")
KLl = Cost.assign("KL_logistic")
KLt = Cost.assign("KL_tanh")
MSE = Cost.assign("MSE")

# Parameters
hidden_size = 2
visible_size = 4
samples = 6

sparse_rate = 0.1
sparse_weight = 0.
decay_weight = 0.

W1,W2,b1,b2 = init_random_uniform(hidden_size,visible_size)

data = np.random.rand(samples,visible_size)

# Test costgrad_sig_sig
print "costgrad_sig_sig:"
def costgrad_sig_sig(W1,W2,b1,b2,hidden_size,sparse_rate,sparse_weight,decay_weight,data):
    # 2 layer SAE
    # Transfer functions: Logistic, logistic
    visible_size = np.shape(data)[1]
    n_samples = np.shape(data)[0]    
    
    data1 = forward_pass_sig(data,W1,b1)
    data2 = forward_pass_sig(data1,W2,b2)    
    data1mean = np.mean(data1,axis=0)
    
    residue = data2-data
    kl = KL_logistic(data1mean,sparse_rate)
    cost = (0.5*np.sum(residue**2)/n_samples+ 
            0.5*decay_weight*np.sum(W1**2)+ 0.5*decay_weight*np.sum(W2**2)+
            sparse_weight*np.sum(kl))    
    
    dkl = dKL_logistic(data1mean,sparse_rate)/n_samples
    delta2 = residue/n_samples
    delta1 = (sparse_weight*np.array([dkl]*n_samples)+
              np.dot(delta2*data2*(1-data2),W2.T))
    W1grad = np.dot(data.T,delta1*data1*(1-data1))+decay_weight*W1
    W2grad = np.dot(data1.T,delta2*data2*(1-data2))+decay_weight*W2
    b1grad = np.sum(delta1*data1*(1-data1),axis=0)
    b2grad = np.sum(delta2*data2*(1-data2),axis=0)    
    
    return cost, W1grad,W2grad,b1grad,b2grad
costgrad0 = costgrad_sig_sig(W1,W2,b1,b2,hidden_size,sparse_rate,sparse_weight,decay_weight,data)

# Layer 0
f0 = {'w_mat':W1,'b_vec' : b1}
Affine0 = Estimator.AffineLayer([f0])
Transfer0 = Estimator.TransferLayer([2], ["logistic"])
DK0 = {'func': DK, 'weight': decay_weight, 'kwargs': {}, 'id': 0}
KL0 = {'func': KLl, 'weight': sparse_weight, 'kwargs': {'sparse_rate': sparse_rate}, 'id':0}
Layer0 = Estimator.Layer(Affine0, Transfer0, [DK0,KL0])

# Layer 1
f1 = {'w_mat':W2,'b_vec' : b2}
Affine1 = Estimator.AffineLayer([f1])
Transfer1 = Estimator.TransferLayer([4], ["logistic"])
DK1 = {'func': DK, 'weight': decay_weight, 'kwargs': {}, 'id': 0}
MSE1 = {'func': MSE, 'weight': 1.0, 'kwargs':{'other': data}, 'id':0}
Layer1 = Estimator.Layer(Affine1, Transfer1, [DK1,MSE1])

# Execute
data_out0, costs_sum0, costs_dict0 = Layer0.feed_forward(data,data_only=False)
data_out1, costs_sum1, costs_dict1 = Layer1.feed_forward(data_out0, data_only=False)
delta1, deriv_w1, deriv_b1 = Layer1.backprop(0,data_in=data_out0,data_out=data_out1, ravelled = False)
delta0, deriv_w0, deriv_b0 = Layer0.backprop(delta1,data_in=data,data_out=data_out0, ravelled = False)
costgrad = (costs_sum0+costs_sum1, deriv_w0, deriv_w1,deriv_b0, deriv_b1)
print compare_list(costgrad0,costgrad)



# Test costgrad_tanh_tanh
print "costgrad_tanh_tanh:"
def costgrad_tanh_tanh(W1,W2,b1,b2,hidden_size,sparse_rate,sparse_weight,decay_weight,data):
    visible_size = np.shape(data)[1]
    n_samples = np.shape(data)[0]     
    
    a2 = forward_pass_tanh(data,W1,b1)
    a3 = forward_pass_tanh(a2,W2,b2)    
    a2mean = np.mean(a2,axis=0)
    residue = a3-data
    kl = KL_tanh(a2mean,sparse_rate)
    cost = (0.5*np.sum(residue**2)/n_samples+
            0.5*decay_weight*(np.sum(W1**2)+np.sum(W2**2))+
            sparse_weight*np.sum(kl))    
    
    dkl = dKL_tanh(a2mean,sparse_rate)
    delta3 = residue*(1-a3**2)
    delta2 = (sparse_weight*np.array([dkl]*n_samples)+
              np.dot(delta3,W2.T))*(1-a2**2)  
    W1grad = np.dot(data.T,delta2)/n_samples+decay_weight*W1
    W2grad = np.dot(a2.T,delta3)/n_samples+decay_weight*W2
    b1grad = np.mean(delta2,axis=0)
    b2grad = np.mean(delta3,axis=0)

    return cost, W1grad,W2grad,b1grad,b2grad
    
costgrad0 = costgrad_tanh_tanh(W1,W2,b1,b2,hidden_size,sparse_rate,sparse_weight,decay_weight,data)

# Layer 0
f0 = {'w_mat':W1,'b_vec' : b1}
Affine0 = Estimator.AffineLayer([f0])
Transfer0 = Estimator.TransferLayer([2], ["tanh"])
DK0 = {'func': DK, 'weight': decay_weight, 'kwargs': {}, 'id': 0}
KL0 = {'func': KLt, 'weight': sparse_weight, 'kwargs': {'sparse_rate': sparse_rate}, 'id':0}
Layer0 = Estimator.Layer(Affine0, Transfer0, [DK0,KL0])

# Layer 1
f1 = {'w_mat':W2,'b_vec' : b2}
Affine1 = Estimator.AffineLayer([f1])
Transfer1 = Estimator.TransferLayer([4], ["tanh"])
DK1 = {'func': DK, 'weight': decay_weight, 'kwargs': {}, 'id': 0}
MSE1 = {'func': MSE, 'weight': 1.0, 'kwargs':{'other': data}, 'id':0}
Layer1 = Estimator.Layer(Affine1, Transfer1, [DK1,MSE1])

# Execute
data_out0, costs_sum0, costs_dict0 = Layer0.feed_forward(data,data_only=False)
data_out1, costs_sum1, costs_dict1 = Layer1.feed_forward(data_out0, data_only=False)
delta1, deriv_w1, deriv_b1 = Layer1.backprop(0,data_in=data_out0,data_out=data_out1, ravelled = False)
delta0, deriv_w0, deriv_b0 = Layer0.backprop(delta1,data_in=data,data_out=data_out0, ravelled = False)
costgrad = (costs_sum0+costs_sum1, deriv_w0, deriv_w1,deriv_b0, deriv_b1)
print compare_list(costgrad0,costgrad)

"""
# Test costgrad_cond_sig_lin
print "costgrad_cond_sig_lin"
D = np.zeros(np.shape(W1))

data = data - 0.5

mask = np.random.randint(0,2,shape(data))
data_masked = copy(data)
data_masked[mask==1] = np.nan
data_masked = np.ma.array(data_masked,mask=mask)

def costgrad_cond_sig_lin(W1,W2,b1,b2,D,hidden_size,sparse_rate,sparse_weight,decay_weight,data,cost_only=False):
    # 2 layer SAE
    # Transfer functions: logistic, linear
    visible_size = np.shape(data)[1]
    n_samples = np.shape(data)[0]
    
    numPer = np.array([np.sum(~data.mask,axis=1)]).T                
    numSeen = np.array([np.sum(~data.mask,axis=0)])
        
    data1 = forward_pass_conditional(data,W1,b1,D)    
    data2 = np.dot(data1,W2)+b2
    data1mean = np.dot((~data.mask).T,data1)/numSeen.T    
    residue = data2-data 
    
    kl = KL_logistic(data1mean,sparse_rate)/visible_size    
    cost1,cost2,cost3,cost4 = 0.5*np.sum(np.sum(residue**2,axis=0)/numSeen), 0.5*decay_weight*np.sum(W1**2)/visible_size, 0.5*decay_weight*np.sum(W2**2), sparse_weight*np.sum(kl)
    
    dkl = dKL_logistic(data1mean,sparse_rate)/visible_size    
    delta2 = residue/numSeen   
    delta1 = np.dot(delta2.filled(0),W2.T) + sparse_weight*np.dot(~data.mask/numSeen,dkl) 
              
    W2grad = np.dot(data1.T,delta2.filled(0)) + decay_weight*W2     
    W1grad = np.dot((data.filled(0)).T,delta1*data1*(1-data1)) + decay_weight*(W1/visible_size)    
    b2grad = np.array([np.sum(delta2,axis=0)]) 
    b1grad = np.array([np.sum(delta1*data1*(1-data1),axis=0)])
    Dgrad = np.dot((~data.mask).T,delta1*data1*(1-data1))
    
    return cost1+cost2+cost3+cost4, W1grad,W2grad,b1grad,b2grad,Dgrad
    
costgrad0 = costgrad_cond_sig_lin(W1,W2,b1,b2,D,hidden_size,sparse_rate,sparse_weight,decay_weight,data_masked)

# Sublayer
layer1 = Estimator.Sublayer("logistic", np.concatenate((W1,D),axis=0), b1, depth = 0, position = 0)
layer2 = Estimator.Sublayer("linear", W2, b2, depth = 1, position = 0)

decay_w = Cost.decay_weighted() 
decay   = Cost.decay()
mse = Cost.mean_squared_error()
spa = Cost.sparsity_KL_logistic()

data0 = np.concatenate((data_masked.filled(0),~data_masked.mask),axis=1)
data1 = layer1.feed_forward(data0)
data2 = layer2.feed_forward(data1)

weights = np.concatenate((np.ones(np.shape(W1)),np.zeros(np.shape(D))),axis=0)

cost =  mse(data2,data_masked.filled(0),data_masked.mask) + decay_weight/visible_size*decay_w(layer1.w_mat,weights) + decay_weight*decay(layer2.w_mat) + sparse_weight*spa(data1,sparse_rate,data_masked.mask)

delta2 = mse.delta(data2,data_masked.filled(0),data_masked.mask)
delta1 = layer2.backprop(delta2,data1,data2) + sparse_weight*spa.delta(data1,sparse_rate,data_masked.mask)

W2grad = layer2.deriv_w(delta2,data1,data2) + decay_weight*decay.deriv_w(layer2.w_mat)
W1_Dgrad = layer1.deriv_w(delta1,data0,data1) + decay_weight/visible_size*decay_w.deriv_w(layer1.w_mat,weights)

W1grad = W1_Dgrad[:visible_size,:]
Dgrad = W1_Dgrad[-visible_size:,:]

b2grad = layer2.deriv_b(delta2,data1,data2)
b1grad = layer1.deriv_b(delta1,data0,data1)

costgrad = (cost,W1grad,W2grad,b1grad,b2grad,Dgrad)

print compare_list(costgrad0,costgrad)
"""

