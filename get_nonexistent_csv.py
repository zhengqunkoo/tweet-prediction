from os import listdir
from os.path import isfile, join
from process_tweets import process_tweets_json

def get_nonexistent_csv(existtype, nonexisttype, filedir):
	'''
	input:
	existtype: filetype that exists
	nonexisttype: filetype that doesn't exist, and want to create
	filedir: directory this function is working in

	writes csv files if not already written
	'''
	onlyfiles = set([f for f in listdir(filedir) if isfile(join(filedir, f))])
	existfiles = set([f for f in onlyfiles if f[-len(existtype):] == existtype])
	nonexistfiles = onlyfiles - existfiles
	for filename in existfiles:
		filename = filename[:-len(existtype)]
		if filename + nonexisttype not in nonexistfiles:
			process_tweets_json(filename, filedir)

if __name__ == '__main__':
	existtype = '.json'
	nonexisttype = '.csv'
	filedir = 'C:/Users/zheng/twitter-files/'
	get_nonexistent_csv(existtype, nonexisttype, filedir)