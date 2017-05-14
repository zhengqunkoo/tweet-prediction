from os import listdir
from os.path import isfile, join

ENC = 'utf8'

def combine_files(filename, filetypes, filedir):
	'''
	input:
	filename: name of file to write to
	filetypes: types of file to combine
	filedir: directory that function works in
	'''
	# only write unique lines
	s = set()

	onlyfiles = [f for f in listdir(filedir) if isfile(join(filedir, f))]
	filename = join(filedir, filename)
	for filetype in filetypes:
		nametype = filename + filetype
		if isfile(nametype):
			print('{} exists'.format(nametype))
		else:
			print('{} does not exist, writing new file'.format(nametype))

		for f in onlyfiles:
			with open(filedir + f, 'r', encoding=ENC) as readfile:
				if f[-len(filetype):] == filetype:
					for line in readfile:
						if len(line) >= 4:
							s.add(line)
					print('{} read'.format(filedir + f))

		with open(nametype, 'w', encoding=ENC) as writefile:
			print('writing to file...')
			# sort in alphabetical order
			for line in sorted(s):
				writefile.write(line)
				
		print('{} written'.format(nametype))

if __name__ == '__main__':
	filename = 'combined'
	filetypes = ['.english']
	filedir = 'C:/Users/zheng/twitter-files/'
	combine_files(filename, filetypes, filedir)