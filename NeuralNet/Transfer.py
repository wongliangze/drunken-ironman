import Error

def assign(transfer,location="unknown location"):
	"""
	    :Parameters:
	    transfer: str
	    	Which transfer function to use
	    	examples: "linear","tanh","logistic"
    	location: str
    		Location of layer/sublayer calling this (optional; for error reporting)
	"""
	if transfer == "linear":
		return linear
	elif transfer == "tanh":
		return tanh_sigmoid
	elif transfer == "logistic":
		return logistic_sigmoid
	else:
		raise Error.InitError("Error.assign","Transfer function not defined/recognized in "+location)
