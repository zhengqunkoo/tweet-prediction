from ParseJSON import ParseJSON
import os

class EntitiesFull(object):
	'''
	Note: only writes all values in EntitiesFull. This class is only intended for gensim_word2vec.py
	
	writes entitiesFull from ParseJSON into txt/*.txt, where * is name of .json file

	:para dirname: location of .json files, and no other files
	:para encoding: format to write .txt files
	'''
	def __init__(self, dirname):
		self.dirname = dirname

	def get_dirname(self):
		return self.dirname

	def write_text(self):
		dirname = self.get_dirname()
		keys = [['entitiesFull', 'value']]

		# for all files in txtdirname, excluding directories
		for jsonname in [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]:

			# creates txt directory inside dirname if doesn't exist
			txtdirname = dirname + 'txt/'
			if not os.path.exists(txtdirname):
				os.makedirs(txtdirname)
				print('{} does not exist, creating directory...'.format(txtdirname))

			jsonpath = dirname + jsonname
			pj = ParseJSON(jsonpath, keys)

			txtname = jsonname.split('.')[0] + '.txt'
			txtpath = txtdirname + txtname

			if os.path.exists(txtpath):
				print('{} already exists'.format(txtname))
			else:
				try:
					with open(txtpath, 'wb') as txtfile:
						for full in pj.parse_json():
							# need to extract the values that we passed parse_json,
							# since parse_json yields a list over all the keys we passed into it

							# delimit with spaces, add newline
							full = ' '.join(full[0]) + '\n'
							# convert to bytes to preserve unicode backslashes
							txtfile.write(full.encode('ascii', 'backslashreplace'))

					print('txt/{} written'.format(txtname))

				except Exception as e:
					# if any error encountered in writing txtfile, delete erroneous txtfile
					os.remove(txtpath)
					print('error {}\ndeleted {}'.format(e, txtpath))

	def unique_text(self):
		'''
		write unique liens to another file in the same directory
		'''
		dirname = self.get_dirname()
		txtdirname = dirname + 'txt/'

		# for all files in txtdirname, excluding directories
		for txtname in [f for f in os.listdir(txtdirname) if os.path.isfile(os.path.join(txtdirname, f))]:
			
			# check each file for .txt filetype
			if txtname.split('.')[1] == 'txt':
				
				# custom filetype: .unique
				uniquename = txtname.split('.')[0] + '.unique'
				uniquepath = txtdirname + uniquename
				txtpath = txtdirname + txtname

				if os.path.exists(uniquepath):
					print('{} already exists'.format(uniquepath))
				else:
					uniquelines = set(open(txtpath).readlines())
					try:
						open(uniquepath, 'wb').writelines(uniquelines)
						print('{} written'.format(uniquepath))

					except Exception as e:
						# if any error encountered in writing uniquepath, delete erroneous uniquefile
						os.remove(uniquepath)
						print('error {} occurred\ndeleted {}'.format(e, uniquepath))

if __name__ == '__main__':
	dirnames = ['train/', 'test/']
	for dirname in dirnames:
		EntitiesFull(dirname).write_text()
		# EntitiesFull(dirname).unique_text()