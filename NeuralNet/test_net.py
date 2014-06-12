# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:31:51 2014

@author: Liang Ze
"""

from __future__ import division
import numpy as np
import Core

hidden_size = 2
visible_size = 7
samples = 50

sparse_rate = 0.1
sparse_weight = 0.5
decay_weight = 0.6


Input_data = np.random.rand(samples,visible_size)
Target_data = Input_data


costs0 = [    
    {'name': 'DK',  'weight':decay_weight, 'in_id': 0, 'out_id':0},    
    {'name': 'KL_logistic', 'weight': sparse_weight, 'out_id':0, 'xparams':{'sparse_rate':sparse_rate}},    
]
costs1 = [
    {'name': 'MSE', 'weight': 1., 'out_id': 0},
    {'name': 'DK',  'weight':decay_weight, 'in_id': 0, 'out_id':0},
]

net = Core.Net.init_by_size([[visible_size],[hidden_size],[visible_size]],'logistic',[costs0, costs1])

cost_dict = net.cost_breakdown(Input_data, Target_data)
cost,grad = net.costgrad(Input_data,Target_data, cost_only=False)

params = net.get_free_params()
net.set_free_params(params*0)
new_params = net.get_free_params()


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
W0 = net.Layers[0].get_affine_props('w_mat')[0]
W1 = net.Layers[1].get_affine_props('w_mat')[0]
b0 = net.Layers[0].get_affine_props('b_vec')[0]
b1 = net.Layers[1].get_affine_props('b_vec')[0]
cost0, W0grad, W1grad, b0grad, b1grad = costgrad_sig_sig(W0,W1,b0,b1,hidden_size,sparse_rate,sparse_weight,decay_weight,Input_data)

grad0 = reduce(lambda x,y: x+y,[list(np.ravel(x)) for x in [W0grad, b0grad, W1grad, b1grad] ])

compare = np.array([grad,grad0]).T

