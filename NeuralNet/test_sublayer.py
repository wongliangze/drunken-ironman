# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:00:59 2014

@author: Liang Ze
"""
import numpy as np
from Estimator import Sublayer
import Transfer
from Cost import decay, sparsity_KL_logistic, mean_squared_error

dc = decay()
kl = sparsity_KL_logistic()
ms = mean_squared_error()

size_0 = 10
size_1 = 5
samples = 50

w_mat = np.random.rand(size_0,size_1)
b_vec = np.zeros(size_1)
data = np.random.rand(samples,size_0)

for transfer in Transfer.transfer_list:
    print transfer
    mysublayer = Sublayer(transfer,w_mat,b_vec, depth=0, position = 0)
    
    output = mysublayer.feed_forward(data)
    residue = output/2
    
    delta = mysublayer.backprop(residue,data,output)
    
    deriv_w = mysublayer.deriv_w(residue,data,output)
    deriv_b = mysublayer.deriv_b(residue,data,output)
    
    dc(mysublayer.w_mat)
    dc.deriv_w(mysublayer.w_mat)
    dc.deriv_b()
    
    if transfer == "logistic":
        kl(output,0.1,)
        kl.delta(output,0.1,)
    
    ms(output,residue,)
    ms.delta(output,residue,)
    
    
    
