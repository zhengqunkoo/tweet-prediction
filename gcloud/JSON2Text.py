"""
Defines functions to handle metadata
"""
import json


def parse_test_case(test_case):
	"""
	Parses a JSON file and yields 2 objects:
	tweet_id
	an initial string for predicting on
	"""
	print(test_case)
	with open("{}.txt".format(test_case), 'w') as f, open(test_case, 'r') as jf:
		for obj in json.load(jf):
			user = obj["user"]
			entities_shortened = obj["entitiesShortened"]
			inputs = []
			first_item = None
			for item in entities_shortened:
				if not first_item:
					first_item = item["value"]
				if item["type"] == "userMention":
					inputs.append("\1@"+item["value"]+"\1")
				elif item["type"] == "hashtag":
					inputs.append("\2#"+item["value"]+"\2")
				elif item["type"] == "url":
					inputs.append("\3<link>\3")
				else:
					inputs.append(item["value"])
			f.write("".join(inputs)+"\t"+first_item)


if __name__ == "__main__":
	import sys
	print("Starting training...")
	if len(sys.argv) >= 2:
		for i in sys.argv[1:]:
			parse_test_case(i)
	else:
		print("Usage: %s [json files]"%sys.argv[0])

