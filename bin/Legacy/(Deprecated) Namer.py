

def name(location):
	if location == None:
		return "unknown"
	else:
		return str(location)

def sublayer_name(depth,position):
	return "Layer " + name(depth) + ", Sublayer " + name(position)