def char2words(char_sequence,filename):
	"""
	A quick and dirty example of how to enumerate all 
        possibilities in the path
	"""
	with open(filename) as words:
		possibilities = [[] for c in char_sequence] 
		for word in words:
			for i in range(len(char_sequence)):
				if word[0] == char_sequence[i]:
					possibilities[i].append(word)
		return possibilities	
