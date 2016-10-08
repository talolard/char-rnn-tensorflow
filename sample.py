from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import os
from six.moves import cPickle as pickle
from six import text_type

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--init_from', type=str,
                    default='checkpoints', help="checkpoint file or directory to intialize from, if directory the most recent checkpoint is used, directory must have vocab and config files")
parser.add_argument('-n', type=int, default=500,
                   help="length of desired sample")
parser.add_argument('--prime', type=text_type, default=u'',
                   help="primer for generation")
parser.add_argument('--temperature', type=float, default=1,
                   help="sampling temperature")
args = parser.parse_args()

assert os.path.exists(args.init_from),"{} is not a file or directory".format(args.init_from)

if os.path.isdir(args.init_from):
    parent_dir = args.init_from
else:
    parent_dir = os.path.dirname(args.init_from)

config_file = os.path.join(parent_dir, "config.pkl")
vocab_file = os.path.join(parent_dir, "vocab.pkl")

assert os.path.isfile(config_file), "config.pkl does not exist in directory {}".format(parent_dir)
assert os.path.isfile(vocab_file), "vocab.pkl does not exist in directory {}".format(parent_dir)

with open(config_file, 'rb') as f:
    saved_args = pickle.load(f)

with open(vocab_file, 'rb') as f:
    saved_vocab = pickle.load(f)

if os.path.isdir(args.init_from):
    checkpoint = tf.train.latest_checkpoint(parent_dir)
    assert checkpoint, "no checkpoint in directory {}".format(init_from)
else:
    checkpoint = args.init_from

model = Model(saved_args, sample=True)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    if args.init_from is not None:
        try:
            saver.restore(sess, checkpoint)
        except ValueError:
            print("{} is not a valid checkpoint".format(checkpoint))
    print(model.sample(sess, saved_vocab, args.n, args.prime, args.temperature))
