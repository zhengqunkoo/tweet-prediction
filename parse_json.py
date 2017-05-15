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
		
		outputs tuple of concatenated values from each keymatch, in order
		'''
		values = []
		for entity in datum:
			if entity['type'] == keymatch:
				values.append(entity['value'])
		return tuple(values)

	def parse_json(self):
		'''
		:para filename: .json filename

		outputs dictionary, where key is tuple of shortened letter entities,
		and value is tuple of full word entities
		'''
		filename = self.get_filename()
		with open(filename) as f:
			shortened_to_full = {}
			for datum in ijson.items(f, 'item'):
				full = self.values_from_datum(datum['entitiesFull'], 'word')
				shortened = self.values_from_datum(datum['entitiesShortened'], 'letter')
				shortened_to_full[shortened] = full

			return shortened_to_full

if __name__ == '__main__':
	for filename in ['example_training_data.json']:
		pj = ParseJSON(filename)
		pprint(pj.parse_json())