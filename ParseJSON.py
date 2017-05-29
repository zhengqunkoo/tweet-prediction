import ijson
from pprint import pprint
from re import sub

class ParseJSON(object):
	def __init__(self, filepath, all_keys, replace_types):
		"""
		:param filepath: .json filepath, where the file is formatted as a list of JSON objects
		:param all_keys: list of lists, each list contains full sequence of keys to extract value from dictionary

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

				if 'type' in json_obj:
					json_type = json_obj['type']
					if json_type in replace_types:
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
	# basic example
	filepath = 'train/tmlc1-training-010.json'
	all_keys = [['id'],['user'],['created'],['media','url'],['isReply'],['quotedText'],['entitiesFull', 'value'],['entitiesShortened', 'value']]
	replace_types = {'hashtag':'\31', 'userMention':'\32', 'number':'\33', 'url':'\34', 'punctuation':'\35', 'emoji':'\36'}
	pj = ParseJSON(filepath, all_keys, replace_types)
	for values in pj.parse_json():
		values = [' '.join(value) if type(value) == list else str(value) for value in values]
		values = '\t'.join(values)
		
		# always replace all '\n' in value with '\37'
		# so that when filepointer.read(), doesn't cut off string midway
		if '\n' in values:
			values = values.replace('\n', '\37')
			values = values.encode('ascii', 'backslashreplace')

	print('====================================')

	# manipulate
	filepath = 'example_training_data.json'
	all_keys = [['entitiesFull', 'type'], ['entitiesFull', 'value']]
	pj = ParseJSON(filepath, all_keys, replace_types)
	pprint(list(pj.filter_keyvaluepair_by_key('word')))