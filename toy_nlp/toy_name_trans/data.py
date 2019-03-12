
CORPUS = "corpus.txt"

def data_generator(corpus):
    with open(corpus) as corpusFile:
        for line in corpusFile:
            line = line.strip().split(' ')
            if len(line) == 1: continue
            yield (line)


