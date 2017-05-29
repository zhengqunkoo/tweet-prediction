from ParseJSON import ParseJSON
import os

class JSON2Text(object):
	"""
	:param dirname: location of .json files, and no other files
	:param 
	"""

	def __init__(self, dirname, replace_types):
		self.dirname = dirname
		self.replace_types = replace_types
		# keys hardcoded to be all possible keys
		# with hardcode, can reliably edit values of keys later
		self.keys = [['id'],['user'],['created'],['media','url'],['isReply'],['quotedText'],['entitiesFull', 'value'],['entitiesShortened', 'value']]
		print('writing with keys: {}'.format(self.keys))

	def write_text(self, rewrite=False):
		
		"""
		writes the values yielded from ParseJSON into txt/*.txt, where * is name of .json file
		values yielded from ParseJSON depends on self.keys
		if a value has multiple words, the words are separated by ' '.
		if multiple keys, values of different keys are separated by '\t'. (assumes tabs are not in tweets)
		each tweet is separated by '\n'.

		:param rewrite: if True, rewrites existing files with same name in the same directory
		"""
		dirname = self.dirname
		keys = self.keys

		# for all files in txtdirname, excluding directories
		for jsonname in [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]:

			# creates txt directory inside dirname if doesn't exist
			txtdirname = dirname + 'txt/'
			if not os.path.exists(txtdirname):
				os.makedirs(txtdirname)
				print('{} does not exist, creating directory...'.format(txtdirname))

			# creates ParseJSON generator
			jsonpath = dirname + jsonname
			pj = ParseJSON(jsonpath, keys, replace_types)

			txtname = jsonname.split('.')[0] + '.txt'
			txtpath = txtdirname + txtname
			if rewrite == False and os.path.exists(txtpath):
				print('{} already exists'.format(txtname))
			else:
				try:
					with open(txtpath, 'wb') as txtfile:
						for values in pj.parse_json():
							# delimit each value with spaces if multiple elements
							# if value is False or None, typecast to string
							values = [' '.join(value) if type(value) == list else str(value) for value in values]

							# separate values by '\t'
							values = '\t'.join(values)

							# always replace all '\n' in value with '\37'
							# so that when filepointer.read(), doesn't cut off string midway
							values = values.replace('\n', '\37')
							# append '\n'
							values = (values + '\n').encode('ascii', 'backslashreplace')
							
							txtfile.write(values)

					print('txt/{} written'.format(txtname))

				except Exception as e:
					# if any error encountered in writing txtfile, delete erroneous txtfile
					os.remove(txtpath)
					print('error {}\ndeleted {}'.format(e, txtpath))

	def unique_text(self, rewrite=False, printlines=0):
		"""
		write unique lines to .unique file in the same directory

		lines are unique based only on their 'user' and 'entitiesFull' keys
		meaning, if more than one line has the same 'user' and 'entitiesFull', the all these lines are considered duplicates
		and only one of the lines will be included in the .unique file

		:param rewrite: if True, rewrites files with same name in the same directory
		:param printlines: prints bytes as string to show you that bytes were written correctly

		calling set() on lines already shuffles the lines, no need to shuffle in model.fit_generator(), in character_rnn.py
		"""
		dirname = self.dirname
		txtdirname = dirname + 'txt/'

		# for all files in txtdirname, excluding directories
		for txtname in [f for f in os.listdir(txtdirname) if os.path.isfile(os.path.join(txtdirname, f))]:
			
			# check each file for .txt filetype
			if txtname.split('.')[1] == 'txt':
				
				# custom filetype: .unique
				uniquename = txtname.split('.')[0] + '.unique'
				uniquepath = txtdirname + uniquename
				txtpath = txtdirname + txtname

				if rewrite == False and os.path.exists(uniquepath):
					print('{} already exists'.format(uniquepath))
				else:
					uniquelines = {}
					for line in open(txtpath, 'rb'):
						# split by tab in bytes
						keys = line.split('\t'.encode('ascii', 'backslashreplace'))
						# 'user' and 'entitiesFull' are hardcoded at these indeces, according to self.keys
						user, entitiesFull = keys[1], keys[6]
						# append newline in bytes
						uniquelines[(user, entitiesFull)] = line
					try:
						with open(uniquepath, 'wb') as f:
							printcount = 0
							for value in uniquelines.values():
								f.write(value)
								if printcount < printlines:
									print(value)
									printcount += 1
						print('{} written'.format(uniquepath))

					except Exception as e:
						# if any error encountered in writing uniquepath, delete erroneous uniquefile
						os.remove(uniquepath)
						print('error {} occurred\ndeleted {}'.format(e, uniquepath))

if __name__ == '__main__':
	dirnames = ['train/', 'test/']
	replace_types = {'number':'\33', 'url':'\34', 'punctuation':'\35', 'emoji':'\36'}
	# set rewrite = True to rewrite all existing .txt and .unique files
	rewrite = True
	for dirname in dirnames:
		j2t = JSON2Text(dirname, replace_types)
		#j2t.write_text(rewrite=rewrite)
		j2t.unique_text(rewrite=rewrite, printlines=1)