# -*- coding: utf-8 -*-
"""
Created on Thu Jun 05 15:30:13 2014

@author: Liang Ze
"""

import Core
import numpy as np

block_sizes_in = [5,5]
block_sizes_out = [3,3]
transfers = ['tanh','logistic']
costs = [
    {'name': 'MSE', 'weight': 1, 'out_id': 1},
    {'name': 'DK',  'weight':0.5, 'in_id': 1, 'out_id':1},
    {'name': 'KL_tanh', 'weight': 0.3, 'out_id':0, 'xparams':{'sparse_rate':0.1}},
    {'name': 'KL_logistic', 'weight': 0.3, 'out_id':1, 'xparams':{'sparse_rate':0.1}},    
]

layer = Core.Layer.init_by_size(block_sizes_in, block_sizes_out,transfers,costs, layer_id = 0)                     

data_in = np.random.rand(3,10)
data_out, cost_sum, cost_dict = layer.feed_forward(data_in,data_only=False)
target_data = data_out/2
prev_delta, deriv_w, deriv_b = layer.backprop(0, data_in, data_out, ravelled = False, target_data = target_data)
prev_delta, ravelled_grads = layer.backprop(0, data_in, data_out, target_data = target_data)
