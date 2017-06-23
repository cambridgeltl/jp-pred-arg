from collections import Counter
class Vocab:

    def __init__(self, init_words=[], unk_token = '<UNK>', unk_threshold=1):
        self.word2int = {}
        self.int2word = []
        self.frozen = False
        self.add(unk_token)
        if len(init_words) > 0:
            self.build(init_words, unk_threshold)

    def add(self, elem):
        assert not self.frozen
        if elem not in self.word2int:
            self.int2word.append(elem)
            self.word2int[elem] = len(self) - 1

    def build(self, words, unk_threshold=1):
        counter = Counter()
        for w in words:
            counter[w] += 1
        for w,c in counter.items():
            if c >= unk_threshold:
                self.add(w)

    def elems(self):
        return self.int2word

    def freeze(self):
        self.frozen = True

    def index(self, elem):
        if self.frozen and elem not in self.word2int:
            return 0
        else:
            assert elem in self.word2int
            return self.word2int[elem]

    def word(self, index):
        assert index < len(self.int2word)
        return self.int2word[index]

    def __len__(self):
        return len(self.int2word)
