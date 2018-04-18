import argparse
import os
import pickle
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from utils.data_loader import get_loader
from utils.model import EncoderCNN, DecoderRNN
from utils.build_vocab import Vocabulary
from utils.logger import Logger


class CaptionGenerator(object):
    def __init__(self, _args):
        self.args = _args
        self.min_loss = 100000
        self.__init_model_path()
        self.transform = self.__init_transform()
        self.vocab = self.__init_vocab()
        self.train_data_loader = self.__init_data_loader(self.args.train_images_json)
        self.val_data_loader = self.__init_data_loader(self.args.val_images_json)
        self.encoder = self.__init_encoder()
        self.decoder = self.__init_decoder()
        self.criterion = self.__init_criterion()
        self.optimizer = self.__init_optimizer()
        self.scheduler = self.__init_scheduler()
        self.logger = self.__init_logger()

    def train(self):
        # Train the Models
        for epoch in range(args.num_epochs):
            train_loss = self.__epoch_train()
            val_loss = self.__epoch_val()
            self.scheduler.step(val_loss)
            print("[{}] Epoch-{} train loss:{} - val loss:{}".format(self.__get_now(), epoch, train_loss, val_loss))
            self.__save_model(val_loss, self.args.saved_model_name)
            self.__log(train_loss, val_loss, epoch)

    def __epoch_train(self):
        train_loss = 0
        for i, (images, captions, _, lengths) in enumerate(self.train_data_loader):
            self.encoder.train()
            images = self.__to_var(images, volatile=True)
            captions = self.__to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, Backward and Optimize
            self.decoder.zero_grad()
            self.encoder.zero_grad()
            features = self.encoder(images)
            outputs = self.decoder(features, captions, lengths)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data[0]
        return train_loss

    def __epoch_val(self):
        val_loss = 0
        for i, (images, captions, _, lengths) in enumerate(self.val_data_loader):
            self.encoder.eval()
            images = self.__to_var(images, volatile=True)
            captions = self.__to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, Backward and Optimize
            features = self.encoder(images)
            outputs = self.decoder(features, captions, lengths)
            loss = self.criterion(outputs, targets)
            val_loss += loss.data[0]
        return val_loss

    def __log(self, train_loss, val_loss, epoch):
        info = {
            'train loss': train_loss,
            'val loss': val_loss
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)

        # (2) Log values and gradients of the parameters (histogram)
        # for tag, value in net.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.histo_summary(tag, to_np(value), step+1)
        #     logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)
        #
        # # (3) Log the images
        # info = {
        #     'images': to_np(images.view(-1, 28, 28)[:10])
        # }
        #
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, step+1)

    def __init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize(args.resize),
            transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __init_vocab(self):
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __init_data_loader(self, images_json):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 images_json=images_json,
                                 vocabulary=self.vocab,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 shuffle=True,
                                 num_workers=self.args.num_workers)
        return data_loader

    def __init_encoder(self):
        encoder = EncoderCNN(self.args.embed_size)
        if self.args.cuda:
            encoder.cuda()
        return encoder

    def __init_decoder(self):
        decoder = DecoderRNN(self.args.embed_size, self.args.hidden_size,
                             len(self.vocab), self.args.num_layers)
        if self.args.cuda:
            decoder.cuda()
        return decoder

    def __init_criterion(self):
        return nn.CrossEntropyLoss()

    def __init_optimizer(self):
        params = list(self.decoder.parameters()) \
                 + list(self.encoder.parameters()) \
                 + list(self.encoder.bn.parameters())
        return torch.optim.Adam(params=params, lr=args.learning_rate)

    def __init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        return scheduler

    def __save_model(self, loss, file_name):
        if loss < self.min_loss:
            print("Saved Model in {}".format(file_name))
            torch.save({'encoder': self.encoder.state_dict(),
                        'decoder': self.decoder.state_dict(),
                        'best_loss': loss,
                        'optimizer': self.optimizer.state_dict()},
                       file_name)
            self.min_loss = loss

    def __to_var(self, x, volatile=False):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def __get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def __get_now(self):
        return str(time.strftime('%m%d-%H:%M:%S', time.gmtime()))

    def __init_logger(self):
        logger = Logger(os.path.join(self.args.log_dir, self.__get_now()))
        return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resize image')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='directory for images')
    parser.add_argument('--caption_json', type=str, default='./data/captions.json',
                        help='path for captions')
    parser.add_argument('--train_images_json', type=str, default='./data/train_files.json',
                        help='path for train files')
    parser.add_argument('--val_images_json', type=str, default='./data/val_files.json',
                        help='path for val files')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='The path for tensorboard.')
    parser.add_argument('--saved_model_name', type=str, default='val')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    model = CaptionGenerator(args)
    model.train()
