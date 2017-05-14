import json

def values_from_datum(datum, keymatch):
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

def parse_json(filename):
	'''
	:para filename: .json filename

	outputs dictionary, where key is tuple of shortened letter entities,
	and value is tuple of full word entities
	'''
	with open('filename') as f:
		shortened_to_full = {}
		data = json.load(f)

		for datum in data:
			full = values_from_datum(datum['entitiesFull'], 'word')
			shortened = values_from_datum(datum['entitiesShortened'], 'letter')
			shortened_to_full[shortened] = full

		return all_full, all_shortened