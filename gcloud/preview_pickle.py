import pickle
import sys
with open(sys.argv[1], 'rb') as out:
	try:
		while True:
			print(pickle.load(out))
	except EOFError:
		pass

