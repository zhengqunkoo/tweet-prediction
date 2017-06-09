from requests import post
from requests.auth import HTTPBasicAuth
import os
from json import dump

def make_submissions(domain, EMAIL, AUTHKEY, id2submissions, results_path, results_file):
	"""
	# ONLY MAKES ONE REQUEST: ALL TWEETS IN ONE JSON OBJECT

	:param domain: string, URL that links to a directory
	:param EMAIL: email as username for HTTPBasicAuth
	:param AUTHKEY: authkey as password for HTTPBasicAuth

	:param id2submissions: list of dicts. dict format: key: tweet id, value: submissions, words only
	:param results_path: folder to write results_file to
	:param results_file: full name of file to write results to
	"""
	errors = []

	# create directory if not exist
	if not os.path.exists(results_path):
		os.makedirs(results_path)

	try:
		contents = post(domain, auth=HTTPBasicAuth(EMAIL, AUTHKEY), json=id2submissions)
	# log unknown errors
	except Exception as e:
		errors.append(domain + '\t' + e)

	# check if contents show success or failure
	if contents:
		with open(results_path + results_file, 'w') as f:
			dump(contents.json(), f)
	# log content errors
	else:
		errors.append(id2submissions + '\tunable to get contents')

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
	import sys
	from metadata_preproc import test_model_twitter
	from ParseJSON import ParseJSON
	import pickle
	import os
	# CHANGE THIS TO SUBMISSIONS WHEN READY
	domain = 'http://challenges.tmlc1.unpossib.ly/api/submissions'
	EMAIL = 'zhengqun.koo@gmail.com'
	AUTHKEY = '69072a84e36a942c33a3ff678b6f23a4'
	results_path = 'results/'
	results_file = 'results.txt'

	if sys.argv[1] == 'submit' and len(sys.argv) == 3:
		id2submissions = {}
		for file in sys.argv[2:]:
			print(file)
			with open(file, 'rb') as f:
				try:
					while True:
						id2submission = pickle.load(f)
						if id2submission:
							for tweet_id, submission in id2submission.items():
								if tweet_id not in id2submissions:
									id2submissions[tweet_id] = []
								id2submissions[tweet_id].append(submission)
				except EOFError:
					pass

		make_submissions(domain, EMAIL, AUTHKEY, id2submissions, results_path, results_file)

	elif sys.argv[1] == 'split' and len(sys.argv) == 4:
		# split json into n files
		# don't handle deleting old split json files
		# because we only pass the newest split json files into test_model twitter
		print("splitting {} into {}".format(sys.argv[2], sys.argv[3]))
		pj = ParseJSON(sys.argv[2], [], {})
		pj.split_json(sys.argv[3])

	elif sys.argv[1] == 'test' and len(sys.argv) >= 5:
		# if pickle exists, read tweet ids into set, so don't duplicate tweets
		print(sys.argv)
		file = '{}.pickle'.format(sys.argv[2])
		print(file)
		tweet_ids = set()
		if os.path.isfile(file):
			with open(file, 'rb') as f:
				try:
					while True:
						tweet_id = list(pickle.load(f).keys())[0]
						tweet_ids.add(tweet_id)
				except EOFError:
					pass
		# append to pickle
		with open(file, 'ab') as out:
			for prediction in test_model_twitter(tweet_ids, *sys.argv[2:]):
				pickle.dump(prediction, out)
	else:
		print("Usage: %s submit <pathToJson>\nOR\n%s split <pathToJson> <n>\nOR\n%s test <pathToJson> <pathToModel> <k> <j>" % (sys.argv[0], sys.argv[0], sys.argv[0]))
	
