import json

def preview_json():
	# dirty way to get id and entitiesShortened
	# extra comma at the last character of entitiesShortened
	file = 'test/tmlc1-testing-011.json'
	with open('result.txt', 'w', encoding='utf8') as f:
		s = json.loads(open(file).read())
		for j in s:
			f.write('"')
			f.write(j['id'])
			f.write('":[')
			for k in j['entitiesShortened']:
				if k['type'] == 'letter':
					f.write('"')
					f.write(k['value'])
					f.write('",')
			f.write('],\n')

def preview_txt():
	file = 'test/txt/tmlc1-testing-001.txt'
	with open(file, 'rb') as f:
		for s in f:
			s = s.split('\t'.encode('ascii', 'backslashreplace'))
			print(s)

if __name__ == '__main__':
	preview_json()
	# preview_txt()
"""
004.txt
b'FYI: Love should always bring you home last night, tonight, and tomorrow night...unless otherwise instructed lol.'
b'\\u201cDo not go where the path may lead, go instead where there is no path and leave a trail.\\u201d\\u2013 Ralph Waldo Emerson'
b'News Alert: Heart disease remains nation\\u2019s leading killer as life expectancy takes its first dip since 1999.\\u2026  '
b'As sea level rises, much of Honolulu and Waikiki vulnerable to groundwater inundation   '
b'You are capable of more than you know...   '
b'Wonder if it was a full marathon and all runners were required to drink a beer after each mile if anyone would cros\\u2026  '

006.txt ################ NOTE THE EXTRA SPACES #####################
b'The holy month of Ramadan is less than three weeks away, are you all excited?      '
b'Sometimes the blessing are not in what he gives, but in what in takes away!'
b' Mubarak, please take a moment to watch, a reminder not only for Muslims but for humanity itself.  '

007.txt
b'Thank you to our amazing law enforcement officers!   '
b'10 to 1 Rachel opens with Nazi salute at meeting of 30 people in DC this weekend. 9pm: "Okay. in 1919 the Treaty of Versailles..."'
b"Just tried watching Saturday Night Live - unwatchable! Totally biased, not funny and the Baldwin impersonation just can't get any worse. Sad"
b'Boo the press if you want. Then imagine what society would be like without a free press.'
b". The same polls that said Hillary would win are the ones saying President Trump's approval rating is low  "
b'According to the ABC/WP poll, among 2016 voters, would beat Hillary Clinton in a rematch -- in the\\u2026  '
"""