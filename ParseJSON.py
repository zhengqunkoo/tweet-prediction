import ijson
from pprint import pprint

class ParseJSON(object):
	def __init__(self, filename):
		self.filename = filename

	def get_filename(self):
		return self.filename

	def values_from_datum(self, datum, keymatch):
		'''
		:para datum: list of dictionaries, each dictionary has 'type' and 'value' keys
		:para keymatch: string to match value corresponding to 'type' key in datum
		
		returns tuple of concatenated values from each keymatch, in order
		'''
		values = []
		for entity in datum:
			if entity['type'] == keymatch:
				values.append(entity['value'])
		return tuple(values)

	def parse_json(self):
		'''
		:para filename: .json filename

		returns 2-tuple:
		tuple of shortened letter entities, and
		tuple of full word entities
		'''
		filename = self.get_filename()
		with open(filename) as f:
			for datum in ijson.items(f, 'item'):
				full = self.values_from_datum(datum['entitiesFull'], 'word')
				shortened = self.values_from_datum(datum['entitiesShortened'], 'letter')
				yield shortened, full

if __name__ == '__main__':
	filename = 'example_training_data.json'
	pj = ParseJSON(filename)
	pprint(pj.parse_json())