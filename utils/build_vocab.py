import pickle
import argparse
from collections import Counter
import json


class JsonReader(object):
    def __init__(self, json):
        self.json = json
        self.data = self.__read_json()

    def __read_json(self):
        with open(self.json, "r") as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        return self.idx2word[id]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold):
    caption_reader = JsonReader(json)
    counter = Counter()

    for each in caption_reader.data:
            counter.update(caption_reader.data[each])

    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for _, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total Vocabulary Size:{}".format(len(vocab)))
    print("Saved path in {}".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='../medical_data/captions.json',
                        help='path for caption file')
    parser.add_argument('--vocab_path', type=str, default='../medical_data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=7,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
