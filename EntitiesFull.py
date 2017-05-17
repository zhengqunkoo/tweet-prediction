from ParseJSON import ParseJSON
import os

class EntitiesFull(object):
	'''
	NOTE: does not write entitiesShortened. This class is only intended for gensim_word2vec.py
	
	writes entitiesFull from ParseJSON into txt/*.txt, where * is name of .json file

	:para dirname: location of .json files, and no other files
	:para encoding: format to write .txt files
	'''
	def __init__(self, dirname, encoding='utf8'):
		self.dirname = dirname
		self.encoding = encoding

	def get_dirname(self):
		return self.dirname

	def write_text(self):
		dirname = self.get_dirname()
		# iterate through all files, but don't iterate through directories
		for jsonname in [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]:
			txtname = jsonname.split('.')[0] + '.txt'
			# creates txt directory inside dirname if doesn't exist
			txtdirname = dirname + 'txt/'
			if not os.path.exists(txtdirname):
				os.makedirs(txtdirname)
				print('{} does not exist, creating directory...'.format(txtdirname))

			pj = ParseJSON(dirname + jsonname)
			if os.path.exists(txtdirname + txtname):
				print('{} already exists'.format(txtname))
			else:
				with open(txtdirname + txtname, 'w', encoding=self.encoding) as txtfile:
					for _, full in pj.parse_json():
						txtfile.write(' '.join(full))
						txtfile.write('\n')
				print('txt/{} written'.format(txtname))

	def unique_text(self):
		'''
		write unique liens to another file in the same directory
		'''
		dirname = self.get_dirname()
		txtdirname = dirname + 'txt/'
		for txtname in [f for f in os.listdir(txtdirname) if os.path.isfile(os.path.join(txtdirname, f))]:
			# check for .txt extension
			if txtname.split('.')[1] == 'txt':
				# custom filetype
				uniquename = txtname.split('.')[0] + '.unique'
				if os.path.exists(txtdirname + uniquename):
					print('{} already exists'.format(txtdirname + uniquename))
				else:
					uniquelines = set(open(txtdirname + txtname, encoding=self.encoding).readlines())
					open(txtdirname + uniquename, 'w', encoding=self.encoding).writelines(uniquelines)
					print('{} written'.format(txtdirname + uniquename))

if __name__ == '__main__':
	dirnames = ['train/', 'test/']
	for dirname in dirnames:
		# EntitiesFull(dirname).write_text()
		EntitiesFull(dirname).unique_text()