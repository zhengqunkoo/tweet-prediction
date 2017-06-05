def is_partial_match(s, t):
	"""
	A partial match is case-insensitive,
	must be a Soundex match,
	and have a Levenshtein distance from the full word entity of 0 or 1.
	"""
	s, t = s.upper(), t.upper()
	return soundex(s) == soundex(t) and levenshtein(s, t)

def levenshtein(s, t):
	"""
	returns True if levenshtein distance between s and t is 0 or 1
	meaning, if perfect match, or
	1 insertion / deletion / substitution needed to transform s to t or vice versa

	very ugly code
	"""
	if s == t:
		return True

	lens, lent = len(s), len(t)
	if lens - 1 == lent:
		# test if delete 1 from s => t
		for i in range(lens):
			if t == (s[:i] + s[i+1:]):
				return True
	elif lent - 1 == lens:
		# test if delete 1 from t => s
		for i in range(lent):
			if s == (t[:i] + t[i+1:]):
				return True
	elif lent == lens:
		# test if 1 substitution into t => s
		for i in range(lent):
			if (s[:i] + s[i+1:]) == (t[:i] + t[i+1:]):
				return True
	return False

def soundex(s):
	"""
	convert string s into soundex code
	first letter also converted into number
	reference: https://www.archives.gov/research/census/soundex.html

	no need to pad zeros to soundex code
	soundex code no size limit, so can just keep appending code for long strings
	return list of codes
	"""
	if s == '':
		return s

	codes = ['BFPV','CGJKQSXZ','DT','L','MN','R']	# soundex codes
	cnsnts = 'BFPVCGJKQSXZDTLMNR'					# consonants
	# adjacent double letters treated as one letter
	# adjacent same soundex code treated as one letter
	# find first consonant
	prev_cnsnt = ''
	for ix, ch in enumerate(s):
		if ch in cnsnts:
			prev_cnsnt = ch
			break
	# if first letter is 
	# if no first consonant, return first letter
	if prev_cnsnt == '':
		return [s[0]]
	# if 'HW' separates two consonants with same soundex code, AND
	# 'AEIOUY' do not separate the two consonants, THEN
	# consonant to right not coded

	full_code = []
	# flag has three values: -1, False, True
	flag_to_code = -1
	# let ch be all consonants without same sound as neighbor
	for i, ch in enumerate(s):
		if i == 0 or (ch != s[i-1] and not same_sound(ch, s[i-1], codes)):
			if ch in cnsnts:
				# if flag True 
				if flag_to_code:
					# find soundex code of ch
					for ix, code in enumerate(codes):
						if ch in code:
							break
					full_code.append(ix + 1)

				# reset variables
				prev_cnsnt = ch
				flag_to_code = -1

			elif flag_to_code == False and ch in 'AEIOUY':
				# if previously flagged False, but AEIOUY between consonants, flag True
				flag_to_code = True

			elif flag_to_code == -1 and ch in 'HW':
				# if flag not set to True or False, and HW between consonants, flag False
				flag_to_code = False

	return full_code

def same_sound(a, b, codes):
	"""
	given two chars a, b, and a soundex code,
	return true if a and b have same soundex code,
	return false if different soundex code, or either not in soundex code
	"""
	for code in codes:
		if a in code and b in code:
			return True
	return False

if __name__ == '__main__':
	# this soundex implementation follows standard rules, but:
	# if first letter in soundex not soundex consonant, then not recorded
	# soundex codes can be longer than 4 characters
	tests = ['Gutierrez','Pfister','Jackson','VanDeusen','Tymczak','Ashcraft']
	tests = [x.upper() for x in tests]
	targets = [[2,3,6,2], [1,2,3,6], [2,2,5], [1,5,3,2,5], [3,5,2,2], [2,6,1,3]]

	print('testing soundex...')
	print('errors:')
	for i in range(len(tests)):
		res = soundex(tests[i])
		if res != targets[i]:
			print(tests[i], res)
	print()

	# all possible levenshtein	
	targets = ['Gutierrez', 'Pfister', 'Jackson', 'VanDeusen', 'Tymczak', 'Ashcraft']
	targets = [x.upper() for x in targets]

	insert_tests = [[target[:ix] + '9' + target[ix:] for ix in range(len(target))] for target in targets]
	sub_tests = [[target[:ix] + '9' + target[ix+1:] for ix in range(len(target))] for target in targets]
	delete_tests = [[target[:ix] + target[ix+1:] for ix in range(len(target))] for target in targets]
	
	all_tests = [insert_tests, sub_tests, delete_tests]

	print('testing levenshtein...')
	print('errors:')
	for tests in all_tests:
		for i in range(len(tests)):
			target = targets[i]
			for rest in tests[i]:
				res = levenshtein(target, rest)
				if not res:
					print(target, rest, res)