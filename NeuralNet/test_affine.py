# -*- coding: utf-8 -*-
"""
Created on Tue Jun 03 16:49:59 2014

@author: Liang Ze
"""

import Estimator
import numpy as np

W1 = np.random.rand(5,3)
b1 = 0
W2 = np.random.rand(6,3)
b2 = np.random.rand(3)
W3 = np.random.rand(5,4)
b3 = np.random.rand(4)


merge_by_block = {'in':[[0,2]], 'out':[[0,1]]}

f1 = {'w_mat':W1,'b_vec' : b1}
f2 = {'w_mat':W2, 'b_vec' : b2}
f3 = {'w_mat':W3, 'b_vec' : b3}

FL = Estimator.AffineLayer([f1,f2,f3],merge_by_block)

data_in = np.random.rand(20,FL.size_in)

data_out = FL.feed_forward(data_in)

prev_delta = FL.backprop(data_out)
FL.deriv_b(data_out)
FL.deriv_w(data_out,data_in)