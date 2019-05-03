import keras
from keras.utils import Sequence
from keras.layers import Dense, LSTM, Input, Embedding, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class LineGenerator(Sequence):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        """The batch size per batch."""
        return 

    def __getitem__(self, idx):
        pass

BASE_DIR   = "/home/lee/workspace/projects/toys/toy_nlp/toy_nnlm/"
TRAIN_FILE = BASE_DIR + "PTB dataset/ptb.train.txt"
VALID_FILE = BASE_DIR + "PTB dataset/ptb.valid.txt"
TEST_FILE  = BASE_DIR + "PTB dataset/ptb.test.txt"

def build_word_dictionary():
    words_set = set()
    for path in [TRAIN_FILE, VALID_FILE]:
        with open(path) as file:
            for i, line in enumerate(file):
                if not line.strip(): continue
                _, text = line.strip().split('\t')
                words = text.split(' ')
                words_set.update(words)

    word_to_id, id_to_word = {'PADDING': 0}, ['PADDING']+[None]*len(words_set)
    for id, word in enumerate(sorted(words_set), 1):
        word_to_id[word] = id
        id_to_word[id] = word

    return word_to_id, id_to_word

#word_to_id, id_to_word = build_word_dictionary()

def convert_corpus(path):
    X, Y = [], []
    with open(path, encoding="utf-8") as corpusFile:
        for line in corpusFile:
            if not line.strip(): continue
            label, text = line.strip().split('\t')
            text_ids, label_id = convert_text(text), convert_label(label)
            X.append(text_ids)
            Y.append(label_id)

    return X, Y

VOCAB_SIZE = 100
EMBED_DIM = 30
MAX_INPUT_LEN = 20
hidden_size = min(512, 200)# len(word_to_id) // 4)

def main():
    embedding = Embedding(VOCAB_SIZE, 
                        EMBED_DIM,
                        input_length=MAX_INPUT_LEN)

    input = Input(shape=(MAX_INPUT_LEN,), dtype="int32")
    input_emb = embedding(input)
    # 如果是双向LSTM则需要concat前向和后向的向量
    # 如果是多层的话，需要stack到一个list中
    # 但是多层双向会发生语言泄露的情况
    # 使用RepeatVector和return_sequences的区别
    # https://stackoverflow.com/questions/51749404/how-to-connect-lstm-layers-in-keras-repeatvector-or-return-sequence-true
    input_lstm = LSTM(hidden_size, return_sequences=True)(input_emb)
    # TimeDistributed的作用
    # https://blog.csdn.net/u012193416/article/details/79477220
    output = Dense(VOCAB_SIZE, activation="softmax")(input_lstm)

    model = keras.Model([input], output)
    model.compile(keras.optimizers.Adam(0.001),
                'sparse_categorical_crossentropy', ['acc', ])
    model.summary()

if __name__ == "__main__":
    main()