import argparse
import json
import os
import pickle

import torch
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from utils.data_loader import get_loader
from utils.model import EncoderCNN, DecoderRNN


class Sampler(object):
    def __init__(self, args):
        self.args = args
        self.transform = self.__init_transform()
        self.vocab = self.__init_vocab()
        self.encoder = self.__init_encoder()
        self.decoder = self.__init_decoder()
        self.data_loader = self.__init_data_loader()
        self.__init_result_path()

    def sample(self):
        progress_bar = tqdm(self.data_loader, desc='Sampling')
        results = {}
        for images, captions, images_id, lengths in progress_bar:
            images = self.__to_var(images, volatile=True)

            feature = self.encoder(images)
            sampled_ids = self.decoder.sample(feature)

            # Decode word_ids to words
            for pred, reference, id in zip(sampled_ids, captions, images_id):
                results[id] = {
                    'GT': self.__vec2sent(reference),
                    'Pred': self.__vec2sent(pred)
                }
        self.__save_json(results)
        return results

    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocab.idx2word[word_id]
            if word == '<start>':
                continue
            if word == '<end>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def __init_result_path(self):
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __to_var(self, x, volatile=False):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __init_encoder(self):
        encoder = EncoderCNN(self.args.embed_size)
        encoder.load_state_dict(torch.load(self.args.model_path)['encoder'])

        if self.args.cuda:
            encoder.cuda()
        encoder.eval()
        return encoder

    def __init_decoder(self):
        decoder = DecoderRNN(self.args.embed_size, self.args.hidden_size,
                             len(self.vocab), self.args.num_layers)
        decoder.load_state_dict(torch.load(self.args.model_path)['decoder'])

        if self.args.cuda:
            decoder.cuda()
        return decoder

    def __init_data_loader(self):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 images_json=self.args.images_json,
                                 vocabulary=self.vocab,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 shuffle=False,
                                 num_workers=self.args.num_workers)
        return data_loader

    def __save_json(self, result):
        with open(os.path.join(self.args.result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path of image for generating caption')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--caption_json', type=str, default='./data/captions.json',
                        help='path for captions')
    parser.add_argument('--model_path', type=str, default='./models/train.pkl',
                        help='path for trained model')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--caption_dir', type=str, default='./data/captions.json',
                        help='path for captions')
    parser.add_argument('--images_json', type=str, default='./data/train_files.json',
                        help='path for train files')
    parser.add_argument('--resize', type=int, default=224,
                        help='size for resize image')
    parser.add_argument('--result_path', type=str, default='./results',
                        help='path for trained model')
    parser.add_argument('--result_name', type=str, default='train',
                        help='the filename of the saved result.')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)

    sampler = Sampler(args)
    sampler.sample()
