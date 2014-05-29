from __future__ import division
import numpy as np

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

