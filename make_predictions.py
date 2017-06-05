from requests import post
from requests.auth import HTTPBasicAuth
import os
from json import dump

def make_predictions(domain, EMAIL, AUTHKEY, id2predictions, results_path, results_file):
	"""
	# ONLY HANDLES ONE PREDICTION PER TWEET

	:param domain: string, URL that links to a directory
	:param EMAIL: email as username for HTTPBasicAuth
	:param AUTHKEY: authkey as password for HTTPBasicAuth

	:param id2predictions: list of dicts. dict format: key: tweet id, value: predictions, words only
	:param results_path: folder to write results_file to
	:param results_file: full name of file to write results to
	"""
	errors = []

	# need to wait 60 seconds every 25 files
	request_counter = 0
	max_request_counter = 25
	sleeptime = 60

	# create directory if not exist
	if not os.path.exists(results_path):
		os.makedirs(results_path)

	for id2prediction in id2predictions:
		if request_counter == max_request_counter:
			print('{} URL requests made, sleeping for {} seconds'.format(request_counter, sleeptime))
			request_counter = 0
			sleep(sleeptime)

		try:
			contents = post(domain, auth=HTTPBasicAuth(EMAIL, AUTHKEY), json=id2prediction)
		# log unknown errors
		except Exception as e:
			errors.append(domain + '\t' + e)

		# check if contents show success or failure
		if contents:
			with open(results_path + results_file, 'w') as f:
				dump(contents.json(), f)
		# log content errors
		else:
			errors.append(str(id2prediction.keys()) + '\tunable to get contents')

		request_counter += 1

	print_url_message(errors)

def print_url_message(lst):
	"""
	:param lst: list of URLs with concatenated error message
	prints entire list
	"""
	if lst:
		for url in lst:
			print(url)
		print()

if __name__ == '__main__':
	domain = 'http://challenges.tmlc1.unpossib.ly/api/tests'
	EMAIL = 'zhengqun.koo@gmail.com'
	AUTHKEY = '69072a84e36a942c33a3ff678b6f23a4'

	id2predictions = [{
"rens0erfsao":["p","P","L","C","b","J","C"],
"revfvdonwg0":["R","L","s","t","1","t","S","a","Y","E","s","b","C"],
"rew01x7y9kw":["h","c","y","m","t","t","l","o","i","t","Z","s"],
"rewae4oa51c":["l","b","e","t","m","s","i","y","a"],
"rexihhnvda8":["w","c","g","f","d","y","u","A","t","t","m","f","c","o","e","c"],
"rey02yocn41":["T","i","n","l","i","t","w","g","h"],
"reybkezbfup":["w","d","y","c","r","O","c","a","n","o","w","A","T","R"],
"rf12g43mwap":["w","s","c","h","a","c","a","l","a","t","g","s","s"],
"rf1dz8on4sg":["W","W","s","d","W","c","w","t"],
"rf26akbzhts":["W","w","s","a","t","O","h","a","i","a","R","m"],
"rf2brqagem8":["a","D","l","u","t","o","o","h","p","h","m","t","t","t","e","a"],
"rf8s3v42dc0":["R","W","d","t","f","a","b","o","H","O","s","I","a","l","l","j","g","u","L"],
"rf9bg3np6gw":["S","h","h","i","b","a","Y","t","d","h","a","g","w"],
"rf9lyslj0u8":["i","t","H","a","s","s","p","w","p","a","p","c","a"],
"rfqm6tz3qio":["I","a","b","a","a","l","n","r","I","l","m"],
"4lxmwin":["W","m","v","o","W","S","A","O","D","b","J"],
"ri6eo3lcmww":["s","y","d","i","a","s","e","T","f","u"],
"ri9695pjojk":["n","a","a","t","w","i","m","m","l"]
}]
	results_path = 'results/'
	results_file = 'results.txt'
	make_predictions(domain, EMAIL, AUTHKEY, id2predictions, results_path, results_file)