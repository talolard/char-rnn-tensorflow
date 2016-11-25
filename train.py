from __future__ import print_function
import tensorflow as tf
import argparse
import os
import time
from six.moves import cPickle as pickle
import utils

from utils import DataLoader
from model import Model


def train(args):
    if args.verbose:
        print(vars(args))

    loader = DataLoader(args)

    if args.init_from is not None:
        assert os.path.exists(args.init_from),"{} is not a file or directory".format(args.init_from)

        if os.path.isdir(args.init_from):
            parent_dir = args.init_from
        else:
            parent_dir = os.path.dirname(args.init_from)

        config_file = os.path.join(parent_dir, "config.pkl")
        vocab_file = os.path.join(parent_dir, "vocab.pkl")

        assert os.path.isfile(config_file), "config.pkl does not exist in directory {}".format(parent_dir)
        assert os.path.isfile(vocab_file), "vocab.pkl does not exist in directory {}".format(parent_dir)

        if os.path.isdir(args.init_from):
            checkpoint = tf.train.latest_checkpoint(parent_dir)
            assert checkpoint, "no checkpoint in directory {}".format(init_from)
        else:
            checkpoint = args.init_from

        with open(os.path.join(parent_dir, 'config.pkl'), 'rb') as f:
            saved_args = pickle.load(f)

        assert saved_args.hidden_size == args.hidden_size, "hidden size argument ({}) differs from save ({})" \
                                                            .format(saved_args.hidden_size, args.hidden_size)
        assert saved_args.num_layers == args.num_layers, "number of layers argument ({}) differs from save ({})" \
                                                            .format(saved_args.num_layers, args.num_layers)

        with open(os.path.join(parent_dir, 'vocab.pkl'), 'rb') as f:
            saved_vocab = pickle.load(f)

        assert saved_vocab == loader.vocab, "vocab in data directory differs from save"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    new_config_file = os.path.join(args.save_dir, 'config.pkl')
    new_vocab_file = os.path.join(args.save_dir, 'vocab.pkl')

    if not os.path.exists(new_config_file):
        with open(new_config_file, 'wb') as f:
            pickle.dump(args, f)
    if not os.path.exists(new_vocab_file):
        with open(new_vocab_file, 'wb') as f:
            pickle.dump(loader.vocab, f)

    model = Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        if args.init_from is not None:
            try:
                saver.restore(sess, checkpoint)
            except ValueError:
                print("{} is not a valid checkpoint".format(checkpoint))

            if args.verbose:
                print("initializing from {}".format(checkpoint))

        for e in range(args.num_epochs):
            lr = args.learning_rate * (args.decay_factor ** e)
            init_states_cell_list = utils.get_states_list(model.initial_state_cell, False)
            init_states_cell_dict = utils.get_states_dict(model.initial_state_cell, False)
            init_states_attn_list = utils.get_states_list(model.initial_state_attn, False)
            init_states_attn_dict = utils.get_states_dict(model.initial_state_attn, False)

            for b, (x, y) in enumerate(loader.train):
                final_states_list = utils.get_states_list(model.end_state, False)
                global_step = e * loader.train.num_batches + b
                start = time.time()
                feed = {model.input: x,
                        model.target: y,
                        model.dropout: args.dropout,
                        model.lr: lr,
                        }

                feed.update(init_states_cell_dict)
                feed.update(init_states_attn_dict)

                train_loss,_,cell_states,attn_states= sess.run( [model.cost, model.train_op] + final_states_list + init_states_attn_list, feed)
                init_states_cell_dict =dict(zip(init_states_cell_list, cell_states))
                init_states_attn_dict= dict(zip(init_states_attn_list, attn_states))
                print(len(cell_states))



                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(global_step,
                            args.num_epochs * loader.train.num_batches,
                            e, train_loss, end - start))

                if  False: #global_step % args.save_every == 1 or (e == args.num_epochs - 1 and b == loader.train.num_batches - 1):
                    all_loss = 0
                    val_state = sess.run(model.zero_state)
                    start = time.time()

                    for b, (x, y) in enumerate(loader.val):
                        feed = {model.input: x,
                                model.target: y}
                        state_feed = {pl: s for pl, s in zip(sum(model.start_state, ()), sum(val_state, ()))}
                        feed.update(state_feed)
                        batch_loss, val_state = sess.run([model.cost, model.end_state], feed)
                        all_loss += batch_loss

                    end = time.time()
                    val_loss = all_loss / loader.val.num_batches
                    print("val_loss = {:.3f}, time/val = {:.3f}".format(val_loss, end - start))
                    checkpoint_path = os.path.join(args.save_dir, 'iter_{}-val_{:.3f}.ckpt' \
                                        .format(global_step, val_loss))
                    saver.save(sess, checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='species', help="directory with processed data")
    parser.add_argument('--init_from', type=str,
                        default=None, help="checkpoint file or directory to intialize from, if directory the most recent checkpoint is used")
    parser.add_argument('--save_every', type=int,
                        default=1024, help="batches per save, also reports loss on validation set")
    parser.add_argument('--save_dir', type=str,
                        default='checkpoints', help="directory to save checkpoints and config files")
    parser.add_argument('--num_epochs', type=int,
                        default=128, help="number of epochs to train")
    parser.add_argument('--batch_size', type=int,
                        default=64, help="minibatch size")
    parser.add_argument('--vocab_size', type=int,
                        default=None, help="vocabulary size, defaults to infer from the input")
    parser.add_argument('--seq_length', type=int,
                        default=64, help="sequence length")
    parser.add_argument('--learning_rate', type=float,
                        default=0.002, help="learning rate")
    parser.add_argument('--decay_factor', type=float,
                        default=0.97, help="learning rate decay factor")
    parser.add_argument('--decay_every', type=int,
                        default=1, help="how many epochs between every application of decay factor")
    parser.add_argument('--grad_clip', type=float,
                        default=5, help="maximum value for gradients, set to 0 to remove gradient clipping")
    parser.add_argument('--hidden_size', type=int,
                        default=128, help="size of hidden units in network")
    parser.add_argument('--num_layers', type=int,
                        default=2, help="number of hidden layers in network")
    parser.add_argument('--dropout', type=float,
                        default=0.9, help="dropout keep probability applied to input between lstm layers")
    parser.add_argument('--verbose', action='store_true',
                        help="verbose printing")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_train_args()
    train(args)
