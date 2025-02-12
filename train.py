from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import argparse
from itertools import product
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from lstm import TextGenerationModel
from nltk.corpus import BracketParseCorpusReader
from collections import Counter
import torch.utils.data as data
import numpy as np

################################################################################
class TextData(data.Dataset):
    def __init__(self, sentences, word2idx, idx2word, vocab_size):
        self.sentences = sentences
        self.word2idx = word2idx
        self._vocab_size = vocab_size
        self._data_size = len(sentences)
        self.idx2word = idx2word

    def __getitem__(self, item):
        offset = np.random.randint(0, len(self.sentences))
        sentence = self.sentences[offset]
        sentence_length = len(sentence)

        inputs = [self.word2idx[sentence[i]] for i in range(0, sentence_length-1)]
        targets = [self.word2idx[sentence[i]] for i in range(1, sentence_length)]

        return inputs, targets

    def __len__(self):
        return self._data_size

    def convert_to_string(self, word_ix):
        return ' '.join(self.idx2word[ix] for ix in word_ix)

    @property
    def vocab_size(self):
        return self._vocab_size


def retrieve_data():
    train_data = BracketParseCorpusReader("data", "02-21.10way.clean")
    val_data = BracketParseCorpusReader("data", "22.auto.clean")
    test_data = BracketParseCorpusReader("data", "23.auto.clean")

    train_words = [x.lower() for x in train_data.words()]
    val_words = [x.lower() for x in val_data.words()]
    test_words = [x.lower() for x in test_data.words()]

    all_words = train_words + val_words + test_words

    word_counter = Counter(all_words)

    vocab = ['PAD', 'SOS', 'EOS'] + list(word_counter.keys())
    vocab_size = len(vocab)

    word2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2word = {i: ch for i, ch in enumerate(vocab)}

    train_sents = [[w.lower() for w in sent] for sent in train_data.sents()]
    val_sents = [[w.lower() for w in sent] for sent in val_data.sents()]
    test_sents = [[w.lower() for w in sent] for sent in test_data.sents()]

    train_dataset = TextData(train_sents, word2idx, idx2word, vocab_size)
    val_dataset = TextData(val_sents, word2idx, idx2word, vocab_size)
    test_dataset = TextData(test_sents, word2idx, idx2word, vocab_size)

    return train_dataset, val_dataset, test_dataset


def seq_sampling(model, dataset, seq_length, temp=None, device='cpu'):
    # Only start with a lowercase character:
    dataset.vocab_size
    pivot = torch.Tensor([[random.randint(0, dataset.vocab_size)]]).long().to(device)
    #pivot = torch.randint(dataset.vocab_size, (1, 1), device=device)
    ramblings = [pivot[0, 0].item()]

    h_and_c = None
    for i in range(1, seq_length):
        out, h_and_c = model.forward(pivot, h_and_c)
        if temp is None or temp == 0:
            pivot[0, 0] = out.squeeze().argmax()
        else:
            dist = torch.softmax(out.squeeze()/temp, dim=0)
            pivot[0, 0] = torch.multinomial(dist, 1)
        ramblings.append(pivot[0, 0].item())
    return dataset.convert_to_string(ramblings)

counter= 0
def train(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    dataset, val, test = retrieve_data()

    torch.save(dataset, config.txt_file + '.dataset')

    #dataset = TextDataset(config.txt_file, config.seq_length)

    model = TextGenerationModel(dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers, config.device,
                                1. - config.dropout_keep_prob)

    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)
    accuracies = [0, 1]
    losses = [0, 1]

    for epochs in range(25):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            # print("batch_inputs")
            # print(len(batch_inputs))
            # print(batch_inputs)
            # print("batch_targets")
            # print(len(batch_targets))
            # print(batch_targets)

            if not batch_inputs:
                continue
            device_inputs = torch.stack(batch_inputs, dim=0).to(device)
            device_targets = torch.stack(batch_targets, dim=1).to(device)

            out, _ = model.forward(device_inputs)
            outt = out.transpose(0, 1).transpose(1, 2)
            optimizer.zero_grad()
            loss = criterion.forward(outt, device_targets)
            losses.append(loss.item())
            accuracy = (outt.argmax(dim=1) == device_targets).float().mean()
            accuracies.append(accuracy)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}, LR = {}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        int(config.train_steps), config.batch_size, examples_per_second,
                        accuracies[-1], losses[-1], optimizer.param_groups[-1]['lr']
                ))

            if step % config.sample_every == 0:
                torch.save(model, config.txt_file + '.model')
                with torch.no_grad(), open(config.txt_file + '.generated', 'a') as fp:
                    for length, temp in product([20], [0, 0.5]):
                        text = seq_sampling(model, dataset, length, temp, device)
                        # print(text)
                        file = open("generated.txt", "a")
                        file.write(text)
                        file.write("")
                        file.close()
                        fp.write("epoch: {} ; Accuracy: {} ; {} ; temp: {} ; {}\n".format(epochs, accuracy, temp,  text))

    print('Done training.')

################################################################################
################################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False, default='southpark.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=15, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=3, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=16, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=20000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.95, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=0.5e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5000, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=5000, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)