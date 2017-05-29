import ijson
from pprint import pprint

class ParseJSON(object):
	def __init__(self, filepath, all_keys):
		"""
		:para filepath: .json filepath, where the file is formatted as a list of JSON objects
		:para all_keys: list of lists, each list contains full sequence of keys to extract value from dictionary

		For each tweet, ParseJSON yields a list of values corresponding to all keys,
		where each value is a list of JSON objects (type: list, dictionary, or string)
		"""

		self.filepath = filepath
		self.all_keys = all_keys

	def get_values(self, json_obj, keys):
		"""
		given a json_obj, which may be a dictionary, or a list of dictionaries,
		find the value which matches sequence of dictionary keys
		"""
		for i in range(len(keys)):
			if type(json_obj) == dict:
				# replace URLs in entitiesFull by '\33' (ASCII escape)
				# does NOT replace URLs in media key (you could change it to)
				if 'type' in json_obj and json_obj['type'] == 'url':
					return '\33'
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
		:para key: string to match key

		assumes parse_json yields [[key1, key2, ...], [value1, value2, ...]]
		only yields values with same key as :para key:
		"""
		for result in pj.parse_json():
			keyvaluepair_samekey = filter(lambda x : x[0] == key, zip(*result))
			value_samekey = map(lambda x : x[1], keyvaluepair_samekey)
			yield list(value_samekey)

if __name__ == '__main__':
	# basic example
	filepath = 'example_training_data.json'
	all_keys = [['id'],['user'],['created'],['media','url'],['isReply'],['quotedText'],['entitiesFull', 'value'],['entitiesShortened', 'value']]
	pj = ParseJSON(filepath, all_keys)
	for values in pj.parse_json():
		values = [' '.join(value) if type(value) == list else str(value) for value in values]
		pprint(values)
		values = '\t'.join(values)
		values = values.encode('ascii', 'backslashreplace')
		print(values)
		print()

	print('====================================')

	# manipulate
	all_keys = [['entitiesFull', 'type'], ['entitiesFull', 'value']]
	pj = ParseJSON(filepath, all_keys)
	pprint(list(pj.filter_keyvaluepair_by_key('word')))