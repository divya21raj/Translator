from pickle import dump

# load file into memory
def load_doc(filename):
	file = open(filename, mode='rt', encoding='utf-8')
	text = file.read()
	file.close()
	return text

# Make corresponding pairs of sentences
# eg - ['Poland and Italy may seem like very different countries.', '폴란드와 이탈리아는 매우 다른 나라들처럼 보여진다.']
def makePairs(text1, text2):
	lines1 = text1.strip().split('\n')
	lines2 = text2.strip().split('\n') 
		
	pairs = []
	for i in range(len(lines1)):
		pairs.append([lines1[i].lower(), lines2[i].lower()])
	
	return pairs

# Save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)


# <=======MAIN LOADING LOGIC=======>

enText = load_doc("korean-english-jhe/jhe-koen-dev.en")
koText = load_doc("korean-english-jhe/jhe-koen-dev.ko")
train = makePairs(enText, koText)
save_clean_data(train, "english-korean-train.pkl")

enText = load_doc("korean-english-jhe/jhe-koen-eval.en")
koText = load_doc("korean-english-jhe/jhe-koen-eval.ko")
test = makePairs(enText, koText)
save_clean_data(test, "english-korean-test.pkl")

save_clean_data(train + test, "english-korean-both.pkl")

print(str(len(test)) + " " + str(len(train))+ " " + str(len(train + test)))

# enText = load_doc("korean-english-news-v1/korean-english-park.train/korean-english-park.train.en")
# koText = load_doc("korean-english-news-v1/korean-english-park.train/korean-english-park.train.ko")

# enText = load_doc("korean-english-news-v1/korean-english-park.test/korean-english-park.test.en")
# koText = load_doc("korean-english-news-v1/korean-english-park.test/korean-english-park.test.ko")