
# coding: utf-8

# # NeuralNet Demo
# ### v0.1.0

# In[1]:

from __future__ import division
import numpy as np
from NeuralNet import Core


# We will create an autoencoder with 7 visible units and 2 hidden units. We'll use a random dataset of 50 samples.
# 
# ## Size parameters and datasets
# 
# Since this is an autoencoder, `Target_data` will be the same as `Input_data`. But for your own purposes, you could set them as different datasets.

# In[2]:

visible_size = 7
hidden_size = 2
samples = 50

Input_data = np.random.rand(samples,visible_size)
Target_data = Input_data


# Let's also generate test datasets for cross validation.

# In[3]:

Test_data = np.random.rand(samples,visible_size)
Test_target = Test_data


# ## Network structure
# 
# First, we define the structure of our network.
# 
# Our network will have $n$=2 layers.
# 
# `block_sizes` will be a list of $n$ + 1 = 3 lists. This specifies the number of nodes in each block in each layer. Don't worry about what a block is. For now, each layer only has 1 block.
# 
# `transfers` will be a list of $n$ lists. This specifies the transfer functions for each layer.

# In[4]:

block_sizes = [[visible_size],[hidden_size],[visible_size]]
transfers = [['logistic'],['logistic']]


# ## Cost functions
# 
# Next, we define the cost functions that we want to include.
# 
# Here, `DK` refers to decay weights on the $W$ matrices, `KL_logistic` is the sparsity function, and `MSE` is the mean-squared error. 
# 
# `costs0` will be costs defined on the 1st layer's parameters and output, while `costs1` are costs defined on the 2nd layer's parameters and output.

# In[5]:

sparse_rate = 0.1
sparse_weight = 0.5
decay_weight = 0.6

costs0 = [    
    {'name': 'DK',  'weight':decay_weight, 'in_id': 0, 'out_id':0},    
    {'name': 'KL_logistic', 'weight': sparse_weight, 'out_id':0, 'xparams':{'sparse_rate':sparse_rate}},    
]
costs1 = [
    {'name': 'MSE', 'weight': 1., 'out_id': 0},
    {'name': 'DK',  'weight':decay_weight, 'in_id': 0, 'out_id':0},
]

costs = [costs0, costs1]


# ## Initializing a Neural Net
# 
# With this info, we can initialize our network:

# In[6]:

net = Core.Net.init_by_size(block_sizes,transfers,costs)


# Before training, let's check that the gradient function is correct. If so, we should expect a very small number.

# In[7]:

net.testgrad(Input_data,Target_data)


# ## Training
# 
# Finally, we can train our net.

# In[8]:

progress_list = net.fit(Input_data, Target_data, Test_data = Test_data , Test_target = Test_target , 
                        save_name = 'Test.p', save_iters = 2, total_iters = 10)


# ## Tracking progress
# 
# `Cost` is the value of the cost function evaluated on `Input_data`, while `Score` is the value of the score function (`MSE` by default) evaluated on `Test_data`.
# 
# By returning the output of `net.fit()` to `progress_list`, we can track the progress of training.
# 
# Notice that the training did not converge (`Maximum iterations reached.`).
# 
# We can pass `progress_list` back into `net.fit()` to continue tracking the progress.

# In[9]:

progress_list = net.fit(Input_data, Target_data, Test_data = Test_data , Test_target = Test_target , 
                        save_name = 'Test.p', save_iters = 2, total_iters = 50, progress_list = progress_list)


# `progress_list` stores the breakdown of the cost, so we can monitor how each component changes. We'll use `pandas` to convert `progress_list` into a DataFrame.

# In[10]:

import pandas as pd
progress = pd.DataFrame(progress_list)
progress


# In[11]:

breakdown_cols = [col for col in progress.columns if col not in ['cost', 'epoch','score','time']]
progress[breakdown_cols].plot(kind="area",stacked=False)


# In[12]:

progress[['cost', 'score']].plot()


# From these graphs, we see that the mean-squared-error isn't really going down, which is to be expected, since we fed in random data. These plots are useful in helping you decide how to adjust your weights (`sparse_weight`,`decay_weight` etc.).
# 
# ## Loading and saving
# 
# `NeuralNet.Utils` contains useful functions for loading and save `Net` objects.
# 
# `net.fit()` automatically saves the net to `save_name`, which we set to `Test.p` above. We can reload it and test if the nets are the same.

# In[13]:

from NeuralNet import Utils

net2 = Utils.load_net('Test.p')

print net.score(Input_data,Target_data)
print net2.score(Input_data,Target_data)


# ## That's all for now!
# 
# Future versions will include:
# 
# * Better documentation
# * Default networks for the autoencoder, multimodal fusion, and missing data.
# * Greedy-layerwise pre-training
# * Normalization
# * Additional display functionalities

# In[ ]:



