import ijson
from pprint import pprint
import re

class ParseJSON(object):
	def __init__(self, filepath, all_keys, replace_types):
		"""
		:param filepath: .json filepath, where the file is formatted as a list of JSON objects
		:param all_keys: list of lists, each list contains full sequence of keys to extract value from dictionary
		:param replace_types: map of JSON data types into special characters

		For each tweet, ParseJSON yields a list of values corresponding to all keys,
		where each value is a list of JSON objects (type: list, dictionary, or string)
		"""

		self.filepath = filepath
		self.all_keys = all_keys
		self.replace_types = replace_types

	def get_values(self, json_obj, keys):
		"""
		given a json_obj, which may be a dictionary, or a list of dictionaries,
		find the value which matches sequence of dictionary keys
		"""
		# replace types in entitiesFull by special ASCII chars
		# does NOT replace URLs in media key (because 'media' has no 'type':'url')
		
		replace_types = self.replace_types

		for i in range(len(keys)):
			if type(json_obj) == dict:
				# preprocessing
				# return 1 or 0 for media and isReply
				# return quotedText or 0 for quotedText
				# convert created YYYY-MM-DD HH:MM:SS to YYYYMMDDHHMMSS
				if keys[i] == 'media':
					if json_obj['media']:
						return 1
					return 0
				elif keys[i] == 'isReply':
					if json_obj['isReply']:
						return 1
					return 0
				elif keys[i] == 'quotedText':
					if json_obj['quotedText']:
						# quotedText is unlabelled string
						# use regex to remove all userMentions, hashtags, urls, emojis from quotedText
						# TODO ##############################################################
						# need replace with whitespace? or can replace with nothing??
						text = json_obj['quotedText']
						text = re.sub(r'\s?@\S*\s', ' ', text)
						text = re.sub(r'#\S*\s', ' ', text)
						text = re.sub(r'http\S*', ' ', text)
						myre = re.compile(u'('
										  u'\ud83c[\udf00-\udfff]|'
										  u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
										  u'[\u2600-\u26FF\u2700-\u27BF])+', 
										  re.UNICODE)
						text = myre.sub(r'', text)
						# strip trailing spaces
						return text.strip(' ')
					return 0
				elif keys[i] == 'created':
					return re.sub('[^0-9]', '', json_obj['created'])

				# filter words from entitiesFull
				# such that entitiesFull does not contain any other type other than 'word'
				# assumes entitiesFull is the only key that contains the type 'word'
				# (e.g. entitiesShortened does not contain the type 'word')
				elif keys[i] == 'entitiesFull':
					return [d['value'] for d in json_obj['entitiesFull'] if d['type'] == 'word']

				elif 'type' in json_obj:
					json_type = json_obj['type']
					if json_type in replace_types:
						# instead of returning actual value, return a placeholder, a special ASCII char
						return replace_types[json_type]

				json_obj = json_obj[keys[i]]

			elif type(json_obj) == list:
				# if json_obj is a list of dictionaries, for all dictionaries in the list,
				# match the subsequent sequence of keys (including the current key), return a list of get_values
				return [self.get_values(new_json_obj, keys[i:]) for new_json_obj in json_obj]

		# if json_obj was never a list, then json_obj is now the desired value. return this value
		return json_obj

	def parse_json(self):
		"""
		yields list of values, where each value corresponds to each dictionary key in keys,
		yields over all JSON objects in .json file
		"""
		with open(self.filepath) as f:
			for json_obj in ijson.items(f, 'item'):
				# since the file is a list of json objects, each json_obj currently is a dictionary
				yield [self.get_values(json_obj, keys) for keys in self.all_keys]

	def filter_keyvaluepair_by_key(self, key):
		"""
		:param key: string to match key

		assumes parse_json yields [[key1, key2, ...], [value1, value2, ...]]
		only yields values with same key as :param key:
		"""
		for result in pj.parse_json():
			keyvaluepair_samekey = filter(lambda x : x[0] == key, zip(*result))
			value_samekey = map(lambda x : x[1], keyvaluepair_samekey)
			yield list(value_samekey)

if __name__ == '__main__':
	# example 1
	filepath = 'train/tmlc1-training-010.json'
	all_keys = [['id'],['user'],['created'],['media'],['isReply'],['quotedText'],['entitiesFull', 'value'],['entitiesShortened', 'value']]
	replace_types = {'hashtag':'\31', 'userMention':'\32', 'number':'\33', 'url':'\34', 'punctuation':'\35', 'emoji':'\36'}

	filepath = 'example_training_data.json'
	# all_keys = [['entitiesFull', 'value'], ['entitiesShortened', 'value']]
	all_keys = [['quotedText']]
	pj = ParseJSON(filepath, all_keys, replace_types)
	for i in pj.parse_json():
		for j in i:
			pprint(j)
	# manipulate
	# pprint(list(pj.filter_keyvaluepair_by_key('word')))

	"""
	print('====================================')
	# example 2
	pj = ParseJSON(filepath, all_keys, replace_types)
	for values in pj.parse_json():
		values = [' '.join(value) if type(value) == list else str(value) for value in values]
		values = '\t'.join(values)
		
		# always replace all '\n' in value with '\37'
		# so that when filepointer.read(), doesn't cut off string midway
		if '\n' in values:
			values = values.replace('\n', '\37')
			values = values.encode('ascii', 'backslashreplace')
	"""