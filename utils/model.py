import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, pretrained=True):
        super(EncoderCNN, self).__init__()
        # TODO Extract Image features from CNN based on other models
        resnet = models.resnet152(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.__init_weights()

    def __init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hidden, _ = self.lstm(packed)
        outputs = self.linear(hidden[0])
        return outputs

    def sample(self, features, states=None):
        sampled_ids = np.zeros((np.shape(features)[0], 20))
        inputs = features.unsqueeze(1)
        for i in range(20):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = torch.max(outputs, 1)[1]
            sampled_ids[:, i] = predicted
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids


if __name__ == '__main__':
    # encoder = EncoderCNN(embed_size=256, pretrained=False)
    decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=1800, num_layers=1)
    image = Variable(torch.randn(3, 3, 224, 224))
    # feature = encoder.forward(image)
    feature = Variable(torch.randn(3, 256))
    caption = Variable(torch.ones(3, 20)).long()
    length = np.array([3, 2, 1])
    print(decoder.forward(feature, caption, length))
