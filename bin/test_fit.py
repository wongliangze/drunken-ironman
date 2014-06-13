# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:29:49 2014

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

net.testgrad(Input_data,Target_data)

Test_data = np.random.rand(samples,visible_size)
Test_target = Test_data

progress_list = net.fit(Input_data, Target_data, Test_data = Test_data , Test_target = Test_target , save_iters = 2, total_iters = 50)

import pandas as pd
progress = pd.DataFrame(progress_list)

breakdown_cols = [col for col in progress.columns if col not in ['cost', 'epoch','score','time']]
progress[breakdown_cols].plot(kind="area",stacked=False)