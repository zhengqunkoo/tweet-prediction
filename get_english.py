from os import listdir
from os.path import isfile, join
from textblob import TextBlob

ENC = 'utf8'

def get_english(sourcefiles, sourcetype, desttype, filedir):
	'''
	# previously to write to a combined file, but now writing to many small files, so not needed

	# check last line of english file, if exists
	lastline = ''
	destfile = filedir + destfile + desttype
	if isfile(destfile):
		lastline = open(destfile, 'r', encoding=ENC).readlines()[-1]
		print('last line: {}'.format(lastline))
	else:
		print('{} not found, writing new file'.format(destfile))
	'''
	onlyfiles = [f for f in listdir(filedir) if isfile(join(filedir, f))]
	for filename in sourcefiles:
		if filename[:-len(sourcetype)] + desttype in onlyfiles:
			continue
		filename = filedir + filename
		with open(filename, 'r', encoding=ENC) as textfile, open(filename[:-len(sourcetype)] + desttype, 'w', encoding=ENC) as englishfile:
			print('writing to {}{}'.format(filename[:-len(sourcetype)], desttype))

			lines = textfile.readlines()
			'''
			# skip beyond the first occurence of lastline, if exists
			if lastline:
				lines = lines[lines.index(lastline) + 1:]
				print(lines[:5])
			'''
			length = len(lines)

			for line in lines:
				# TextBlob neededs string length at least 3
				# since string includes '\n', at least 4
				if len(line) >= 4 and TextBlob(line).detect_language() == 'en':
					englishfile.write(line)

		print('{}{} written'.format(filename, desttype))

if __name__ == '__main__':
	filedir = 'C:/Users/zheng/twitter-files/'#064356
	sourcetype = '.csv'
	onlyfiles = set([f for f in listdir(filedir) if isfile(join(filedir, f))])
	sourcefiles = set([f for f in onlyfiles if f[-len(sourcetype):] == sourcetype])
	desttype = '.english'
	get_english(sourcefiles, sourcetype, desttype, filedir)