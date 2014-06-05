# -*- coding: utf-8 -*-
"""
Created on Thu Jun 05 15:30:13 2014

@author: Liang Ze
"""

import Estimator
import Cost
import numpy as np

# Fake data
data_in = np.random.rand(20,11)
data_ref = np.random.rand(20,3)

# Affine Layer Params
W0 = np.random.rand(5,3)
b0 = 0
W1 = np.random.rand(6,3)
b1 = np.random.rand(3)
W2 = np.random.rand(5,4)
b2 = np.random.rand(4)
merge_by_block = {'in':[[0,2]], 'out':[[0,1]]}
f0 = {'w_mat':W0,'b_vec' : b0, 'w_free' : np.random.randint(0,2,np.shape(W0))}
f1 = {'w_mat':W1, 'b_vec' : b1, 'b_free' : np.random.randint(0,2,np.shape(b1))}
f2 = {'w_mat':W2, 'b_vec' : b2}

AffineLayer = Estimator.AffineLayer([f0,f1,f2],merge_by_block)

# Transfer Layer Params
TransferLayer = Estimator.TransferLayer([3,4], ["tanh","logistic"])

# Cost params
DK = Cost.assign("DK")
KL_logistic = Cost.assign("KL_logistic")
KL_tanh = Cost.assign("KL_tanh")
MSE = Cost.assign("MSE")

p0 = {'func': DK, 'weight': 0.5, 'kwargs': {}, 'id': 0}
p1 = {'func': DK, 'weight': 0.1, 'kwargs': {}, 'id': 1}
p2 = {'func': DK, 'weight': 3.1, 'kwargs': {}, 'id': 2}
l0 = {'func': KL_tanh, 'weight': 2.0, 'kwargs': {'sparse_rate': 0.1}, 'id':0 }
l1 = {'func': KL_logistic, 'weight': 1.0, 'kwargs': {'sparse_rate': 0.1}, 'id':1 }
l2 = {'func': MSE, 'weight': 2.3, 'kwargs':{'other': data_ref}, 'id':0}

Costs = [p0,p1,p2,l0,l1,l2]

# Initialize Layer
Layer = Estimator.Layer(AffineLayer,TransferLayer,Costs)

# Run tests
data_out, costs_sum, costs_dict = Layer.feed_forward(data_in, data_only = False)
prev_delta, deriv_w, deriv_b, len_deriv_w, len_deriv_b = Layer.backprop(data_out/2,data_in=data_in,data_out=data_out, ravelled = True)











