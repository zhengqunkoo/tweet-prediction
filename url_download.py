from urllib.request import urlopen
from urllib.parse import urlsplit, urljoin
from urllib.error import HTTPError
import requests
import os
from time import sleep
from json import loads

def url_download(domain, file_sequence, dirpath, EMAIL, AUTHKEY):
	'''
	:para domain: string, URL that links to a directory
	:para file_sequence: list of full filenames, in sequence of file numbering
	:para dirpath: relative path to folder to save files to
	:para EMAIL: email as username for HTTPBasicAuth
	:para AUTHKEY: authkey as password for HTTPBasicAuth

	downloads files in file_sequence from domain, stores files in dirpath
	after downloading, prints list of downloaded files and other errors
	'''
	downloaded = []
	errors = []

	# add '/' to string if needed
	domain = append_slash(domain)
	dirpath = append_slash(dirpath)

	# need to wait 60 seconds every 25 files
	filecounter = 0
	filecounterlimit = 25
	sleeptime = 60

	# create directory if not exist
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

	for file in file_sequence:
		if filecounter == filecounterlimit:
			print('{} URL requests made, sleeping for {} seconds'.format(filecounter, sleeptime))
			filecounter = 0
			sleep(sleeptime)

		# if file exists, don't download
		if os.path.exists(dirpath + file):
			print('{} already exists, continuing...'.format(dirpath + file))
			continue
		print('{} does not exist, downloading...'.format(dirpath + file))

		url = urljoin(domain, file)
		contents = ''
		try:
			contents = urlopen(url).read()
		except HTTPError:
			# try authorization
			with requests.Session() as s:
				s.auth = (EMAIL, AUTHKEY)
				auth = s.post(url)
				r = s.get(url)
				contents = r.content
		# log unknown errors
		except Exception as e:
			errors.apppend(url + '\t' + e)

		# check if contents show success or failure
		if contents:# and loads(contents.decode('utf-8'))['success'] != False:
			downloaded.append(file)
			with open(dirpath + file, 'wb') as f:
				f.write(contents)
		# log content errors
		else:
			errors.append(url + '\tunable to get contents')

		filecounter += 1

	# log results in stdout
	if downloaded:
		print('these files downloaded:')
		print_url_message(downloaded)
	else:
		print('No files downloaded\n')
	print_url_message(errors)

def print_url_message(lst):
	'''
	:para lst: list of URLs with concatenated error message
	prints entire list
	'''
	if lst:
		for url in lst:
			print(url)
		print()

def append_slash(string):
	'''
	:para string: string

	add forward slash to string if slash or backslash does not exist at end of string
	'''
	if string[-1] != '/' and string[-1] != '\\':
		string += '/'
	return string

def get_file_sequence(filename_filetype, start, end):
	'''
	:para domain: string, URL that links to a directory
	:para filename_filetype: dictionary, key is string of filename, value is string of filetype
	:para start: start of file numbering
	:para end: end of file numbering

	concatenates filename, file numbering, and filetype together into a full filename,
	where file numbering ranges from start to end.
	assumes file numbering is in range 01 to 99.

	returns list of full filenames, in sequence of file numbering
	'''
	if not 1 <= start <= end <= 99:
		raise Exception('File numbering wrong. Please ensure 1 <= start <= end <= 99')
	file_sequence = []
	for filename, filetype in filename_filetype.items():
		for i in range(start, end + 1):
			i = str(i)
			# pad
			if len(i) == 1:
				i = '0' + i
			file_sequence.append(filename + i + filetype)
	return file_sequence

if __name__ == '__main__':
	domain = 'http://challenges.tmlc1.unpossib.ly/api/datasets/'
	filename_filetype = {'tmlc1-training-':'.tar.gz', 'tmlc1-testing-':'.tar.gz'}
	file_sequence = get_file_sequence(filename_filetype, 1, 30)
	dirpath = 'downloads/'
	EMAIL = 'zhengqun.koo@gmail.com'
	AUTHKEY = '69072a84e36a942c33a3ff678b6f23a4'
	url_download(domain, file_sequence, dirpath, EMAIL, AUTHKEY)
