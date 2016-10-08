from __future__ import print_function
import argparse
import os
import numpy as np
import codecs
import collections
from six.moves import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str,
                    default='data/on_the_origin_of_species.txt', help="data file in utf-8")
parser.add_argument('--data_dir', type=str,
                    default='species', help="save directory for preprocessed files")
parser.add_argument('--val_frac', type=float,
                    default=0.1, help="fraction of data to use as validation set")
parser.add_argument('--test_frac', type=float,
                    default=0.1, help="fraction of data to use as test set")
parser.add_argument('--verbose', action='store_true',
                    help="verbose printing")
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

vocab_file = os.path.join(args.data_dir, "vocab.pkl")
train_file = os.path.join(args.data_dir, "train.npy")
val_file = os.path.join(args.data_dir, "val.npy")
test_file = os.path.join(args.data_dir, "test.npy")

with codecs.open(args.input_file, 'r', encoding='utf-8') as f:
    data = f.read()

counter = collections.Counter(data)
counts = sorted(counter.items(), key=lambda x: -x[1])

tokens, _ = zip(*counts)
vocab_size = len(tokens)

vocab = dict(zip(tokens, range(vocab_size)))

with open(vocab_file, 'wb') as f:
    pickle.dump(vocab, f)

data_tensor = np.array(list(map(vocab.get, data)))
data_size = data_tensor.shape[0]

val_size = int(args.val_frac * data_size)
test_size = int(args.test_frac * data_size)
train_size = data_size - val_size - test_size

np.save(train_file, data_tensor[:train_size])
np.save(val_file, data_tensor[train_size:train_size + val_size])
np.save(test_file, data_tensor[train_size + val_size:data_size])

if args.verbose:
    print("preprocess done")
    print("vocab size: {}, data size: {}, train size: {}, validation size: {}, test size {}" \
        .format(vocab_size, data_size, train_size, val_size, test_size))
    print("vocab and frequency rank: {}".format(vocab))
