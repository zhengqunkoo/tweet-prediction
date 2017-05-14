'''
either oauth or client needs to be redefined every iteration
otherwise TweetWriter will write 0 tweets!
'''
from nltk.twitter import Streamer, TweetWriter, credsfromfile

def stream(step, limit=2000):
	'''
	gets stream data in english language with few hundred common words tracked
	limit changes according to network capability, start with 2000 limit
	if better network, accelerate change of limit, else, decelerate change of limit

	step size changes change of acceleration

	continuous acceleration
	# 2 * 2010 - 2000 + 10 = 2030
	# 2 * 2030 - 2010 + 10 = 2060
	# 2 * 2060 - 2030 + 10 = 2100

	continuous deceleration
	# 2 * 2010 - 2000 - 20 = 2000
	# 2 * 2000 - 2010 - 20 = 1970
	# 2 * 1970 - 2000 - 20 = 1920

	when limit is near damp_limit, in vicinity, the rate of acceleration / deceleration is reduced
	'''
	limit_log = []

	while True:
		try:
			oauth = credsfromfile()
			client = Streamer(**oauth)

			limit_log.append(limit)
			print(limit_log)

			client.register(TweetWriter(limit=limit))
			client.statuses.filter(languages=['en'], track=['the, i, to, a, and, is, in, it, you, of, for, on, my, that, at, with, me, do, have, just, this, be, so, are, not, was, but, out, up, what, now, new, from, your, like, good, no, get, all, about, we, if, time, as, day, will, one, twitter, how, can, some, an, am, by, going, they, go, or, has, rt, know, today, there, love, more, work, , too, got, he, 2, back, think, did, lol, when, see, really, had, great, off, would, need, here, thanks, been, blog, still, people, who, night, want, why, bit.ly, home, should, well, 3, oh, much, u, then, right, make, last, over, way, can’t, does, getting, watching, 1, its, only, her, post, his, morning, very, she, them, could, first, than, better, after, 2009, tonight, our, again, down, twitpic.com, news, man, 4, im, looking, us, tomorrow, best, into, any, hope, week, nice, show, yes, where, take, check, come, 10, trying, fun, say, working, next, happy, were, even, live, watch, feel, thing, life, little, never, something, bad, free, doing, world, ff.im, 5, video, sure, yeah, bed, let, use, their, look, being, long, done, sleep, before, year, find, awesome, big, un, +, things, ok, another, him, cool, old, ever, help, anyone, made, ready, days, die, other, read, because, two, playing, though, is.gd, house, always, also, listening, maybe, please, wow, haha, having, thank, pretty, game, someone, school, those, snow, twurl.nl, gonna, hey, 7, many, start, wait, while, google, finally, everyone, para, try, 30, god, weekend, most, iphone, stuff, around, music, looks, may, thought, keep, yet, reading, must, which, same, real, follow, bit, hours, might, actually, online, job, friends, said, obama, coffee, hate, hard, soon, tweet, por, making, wish, call, movie, tell, thinking, via, site, 20, facebook, few, found'])
			
			'''
			if abs(damp_limit - limit) <= vicinity:
				limit_log.append(2 * limit_log[-1] - limit_log[-2] + damp_step)
			else:
				limit_log.append(2 * limit_log[-1] - limit_log[-2] + step)
			'''
			limit += step
		except:
			limit -= step
			'''
			if abs(damp_limit - limit) <= vicinity:
				limit_log.append(2 * limit_log[-1] - limit_log[-2] - 2*damp_step)
			else:
				limit_log.append(2 * limit_log[-1] - limit_log[-2] - 2*step)
			'''
			
if __name__ == '__main__':
	step = 20
	stream(step)