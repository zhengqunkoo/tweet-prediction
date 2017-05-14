import json
import pprint
import re

ENC = 'utf8'

def process_tweets_json(filename, filedir):
	'''
	filedir: directory function works in
	filename: name of .json file

	convert json file to csv file
	'''
	filename = filedir + filename
	with open(filename + '.json', 'r', encoding=ENC) as jsonfile, open(filename + '.csv', 'w', encoding=ENC) as csvfile:
		for line in jsonfile:
			try:
				# strip extra characters
				text = json.loads(line)['text']
				# if tweet contains '…', remove word including '…'
				text = re.sub(r' \S*?…\S*\n', '\n', text)
				# since hashtags (# character) are usually english words (or some mangled english words),
				# only remove hashtag character, but keep the hashtag words
				text = re.sub('#', '', text)
				# remove twitter retweet and user handles
				text = re.sub(r'[\'"]?RT\s@\S*\s', " ", text)
				# remove all user handles (@ character)
				# finds '@', finds ' ', removes all text in between
				# different delimiters, tweets could use '\n' instead of ' '
				text = re.sub(r'\s?@\S*\s', " ", text)
				# remove URLs
				# since links usually at end of tweet, just remove all text after link
				# if only remove link, certain words about the link might contaminate previous words
				# definition of urls
				urls = ['http']
				for url in urls:
					text = re.sub(r'http\S*', ' ', text)
				# replace all contiguous whitespace with spaces, and remove trailing whitespaces
				text = re.sub(r'\s+', ' ', text).strip()
				# remove all quotations at start or end of text
				text = re.sub(r'^[\'"].*[\'"]$', '', text)
				# if text not empty
				if len(text) >= 4:
					csvfile.write(text+'\n')
			except:
				# catch and print any text that causes Exceptions
				print(text)
				return
	print('{}{} written'.format(filename, '.csv'))

if __name__ == '__main__':
	filedir = 'C:/Users/zheng/twitter-files/'
	filename = ''
	process_tweets_json(filename, filedir)