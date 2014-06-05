from __future__ import division
import numpy as np
import operator

# HELPER FUNCTIONS
def getfree(param,free):
    if np.shape(param) == np.shape(free):
        out = list(param[free != 0])        
    else:
        # free is either 1 or 0
        out = list(np.ravel(param)) * free
    return out
def setfree(param,free,new_param):
    if np.shape(param) == np.shape(free):
        param[free != 0] = new_param
    elif free:
        # free is 1
        param[np.ones_like(param) == 1] = np.ravel(new_param)

"""
def getfree_deriv_w(self,delta,data_in):
    return getfree(self.deriv_w(delta,data_in),self.w_free)
def getfree_deriv_b(self,delta):
    return getfree(self.deriv_b(delta), self.b_free)

def getfree_w(self,):
    return getfree(self.w_mat,self.w_free)
def getfree_b(self,):
    return getfree(self.b_vec,self.b_free)

def setfree_w(self,new_w):
    setfree(self.w_mat,self.w_free,new_w)        
def setfree_b(self,new_b):
    setfree(self.b_vec,self.b_free,new_b)
"""        

def merge_blocks(merge_list,num_blocks=None):
    if len(merge_list) == 0:
        return []

    assert len(reduce(operator.add, merge_list)) == len(set(reduce(operator.add, merge_list))), "Merge sets must be disjoint"
    if num_blocks == None:
        num_blocks = max([max(m) for m in merge_list]) + 1

    out = range(num_blocks)
    for merge_set in merge_list:
        for elm in merge_set:
            out[elm] = min(merge_set)

    reduced_index = list(set(out))

    return [reduced_index.index(i) for i in out]
                

def indicator_vec(idx_list,max_idx = None):
    """
    Returns a binary vector of length max_idx with 1 in the positions specified by idx_list, and 0 elsewhere.

    :Parameters:
        idx_list: list of int
            indices that we want to set to 1
        max_idx: int
            length of vector to be returned; defaults to maximum entry of idx_list if not specified
    """
    max_idx = max_idx or max(list(idx_list))
    out = np.zeros(max_idx+1)
    out[idx_list] = 1.
    return out.astype('bool')

def indicator_mat(idx_list_1,idx_list_2,max_idx_1 = None,max_idx_2 = None):
    """
    Returns a binary vector of length max_idx with 1 in the positions specified by idx_list, and 0 elsewhere.

    :Parameters:
        idx_list_1,2: list of int
            indices that we want to set to 1, along dim 1,2
        max_idx_1,2: int
            length of dim 1,2; defaults to maximum entry of idx_list_1,2 if not specified
    """
    max_idx_1 = max_idx_1 or max(list(idx_list_1))    
    dim_1 = np.zeros(max_idx_1+1)
    dim_1[idx_list_1] = 1.

    max_idx_2 = max_idx_2 or max(list(idx_list_2))    
    dim_2 = np.zeros(max_idx_2+1)
    dim_2[idx_list_2] = 1.
    
    return np.outer(dim_1,dim_2).astype('bool')


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

