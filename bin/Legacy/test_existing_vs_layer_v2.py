# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:32:18 2014

@author: Liang Ze
"""

from __future__ import division
import numpy as np
import Core

# Parameters
hidden_size = 2
visible_size = 7
samples = 50

sparse_rate = 0.1
sparse_weight = 0.5
decay_weight = 0.6

data = np.random.rand(samples,visible_size)

costs0 = [    
    {'name': 'DK',  'weight':decay_weight, 'in_id': 0, 'out_id':0},    
    {'name': 'KL_logistic', 'weight': sparse_weight, 'out_id':0, 'xparams':{'sparse_rate':sparse_rate}},    
]
layer0 = Core.Layer.init_by_size([visible_size], [hidden_size],'logistic',costs0, layer_id = 0)                     

costs1 = [
    {'name': 'MSE', 'weight': 1., 'out_id': 0},
    {'name': 'DK',  'weight':decay_weight, 'in_id': 0, 'out_id':0},
]
layer1 = Core.Layer.init_by_size([hidden_size], [visible_size],'logistic',costs1, layer_id = 1)                     

data0, cost_sum0, cost_dict0 = layer0.feed_forward(data,data_only=False,  Target_data = data)
data1, cost_sum1, cost_dict1 = layer1.feed_forward(data0, data_only=False,  Target_data = data)
prev_delta1, deriv_w1, deriv_b1 = layer1.backprop(0, data0, data1, ravelled = False, Target_data = data)
prev_delta0, deriv_w0, deriv_b0 = layer0.backprop(prev_delta1, data, data0, ravelled = False, Target_data = data)
costgrad = [cost_sum0 + cost_sum1, deriv_w0, deriv_w1, deriv_b0, deriv_b1]


def sigmoid(x):
    return 1/(1+np.exp(-x))    
def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))    
    
def forward_pass_sig(data,W,b):        
    return sigmoid(np.dot(data,W) + b)

def KL_logistic(x,sparse_rate):
    return sparse_rate*np.log(sparse_rate/x) + (1-sparse_rate)*np.log((1-sparse_rate)/(1-x))
def dKL_logistic(x,sparse_rate):
    return -sparse_rate/x+(1-sparse_rate)/(1-x)

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
W0 = layer0.get_affine_props('w_mat')[0]
W1 = layer1.get_affine_props('w_mat')[0]
b0 = layer0.get_affine_props('b_vec')[0]
b1 = layer1.get_affine_props('b_vec')[0]
costgrad0 = costgrad_sig_sig(W0,W1,b0,b1,hidden_size,sparse_rate,sparse_weight,decay_weight,data)
