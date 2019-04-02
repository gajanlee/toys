import itertools
import torch

CORPUS = "corpus.txt"
MODE = "E2C"    # English to Chinese

MAX_TOKEN_LEN = 10

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

def nameLine_generator(corpus=CORPUS):
    with open(corpus) as corpusFile:
        for line in corpusFile:
            line = line.strip().split(' ')
            if len(line) == 1: continue
            yield (line)


def name_pairs_generator(mode=MODE):
    for pair in nameLine_generator():
        if mode == "E2C":
            yield (pair[0], pair[1])
        elif mode == "C2E":
            yield (pair[1], pair[0])
        else:
            raise ValueError(mode, " is invalid pair generator mode.")

def zeroPadding(indexes, fillvalue=PAD_token):
    return list(itertools.zip_longest(*indexes, fillvalue=fillvalue))

def binaryMatrix(indexes, value=PAD_token):
    m = []
    for seq in indexes:
        m.append([(0 if token == value else 1) for token in seq])
    return m

def preprocess(word):
    if type(word) is list:
        return map(preprocess, word)
    else:
        return word.lower()

class Vocabulary:

    def __init__(self, name):
        self.name = name

        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.word2index = {}
        self.word2count = {}
        self.num_words = 3

    def addWords(self, words):
        for word in words: self.addWord(word)

    def addWord(self, word):
        word = preprocess(word)
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def indexesFromWords(self, words):
        words = preprocess(words)
        return [SOS_token] + [self.word2index[word] for word in words] + [EOS_token]

    def batch2TrainData(self, pair_batch):
        # len(pair_batch) == batch_size
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        
        input, lengths = self.inputVar(input_batch)
        output, mask, max_target_len = self.outputVar(output_batch)

        return input, lengths, output, mask, max_target_len
    
    def inputVar(self, inputs):
        inputs = preprocess(inputs)
        indexes_batch = [self.indexesFromWords(input) for input in inputs]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths
    
    def outputVar(self, outputs):
        outputs = preprocess(outputs)
        indexes_batch = [self.indexesFromWords(output) for output in outputs]
        max_target_len = max(len(indexes) for indexes in indexes_batch)
        padList = zeroPadding(indexes_batch)
        mask = binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

def getTranslateNameVocabulary():
    voc = Vocabulary("nameTranslator")
    for english, chinese in name_pairs_generator():
       voc.addWords(english)
       voc.addWords(chinese)
    return voc

