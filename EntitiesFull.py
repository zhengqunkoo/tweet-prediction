from ParseJSON import ParseJSON
import os

class EntitiesFull(object):
	'''
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
		for jsonname in os.listdir(dirname):
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
				with open(txtdirname + txtname, 'w+', encoding=self.encoding) as txtfile:
					for _, full in pj.parse_json():
						txtfile.write(' '.join(full))
						txtfile.write('\n')
				print('{} written'.format(txtname))

	def unique_text(self):
		'''
		write unique liens to another file in the same directory
		'''
		dirname = self.get_dirname()
		txtdirname = dirname + 'txt/'
		for txtname in os.listdir(txtdirname):
			uniquelines = set(open(txtname).readlines())
			tmp = open('unique_' + txtname, 'w').writelines(uniquelines)
			tmp.close()
			print('unique_{} written'.format(txtname))

if __name__ == '__main__':
	dirnames = ['train/', 'test/']
	for dirname in dirnames:
		ef = EntitiesFull(dirname).write_text()