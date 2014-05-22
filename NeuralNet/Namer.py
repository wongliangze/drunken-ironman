

def name(input):
	if input == None:
		return "unknown"
	else:
		return str(depth)

def sublayer_name(depth,position):
	return "Layer " + name(depth) + ", Sublayer " + name(position)