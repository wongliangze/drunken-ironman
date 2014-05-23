
def KL(x,sparse_rate):
	assert x*(1.-x) > 0, "Saturated KL computation"
    return sparse_rate*np.log(sparse_rate/x) + (1.-sparse_rate)*np.log((1.-sparse_rate)/(1.-x))

def dKL(x,sparse_rate):
	assert x*(1.-x) > 0, "Saturated dKL computation"
    return -sparse_rate/x+(1.-sparse_rate)/(1.-x)	