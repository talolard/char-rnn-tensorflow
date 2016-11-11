from __future__ import division
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import variable_scope


class Model():

    def __init__(self, args):
        layer_type = rnn_cell.BasicLSTMCell
        layer = layer_type(args.hidden_size, state_is_tuple=True)
        self.dropout = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32), None)
        wrapped = tf.nn.rnn_cell.DropoutWrapper(layer, input_keep_prob=self.dropout)
        self.core = rnn_cell.MultiRNNCell([wrapped] * args.num_layers, state_is_tuple=True)

        self.input = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.target = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.zero_state = self.core.zero_state(args.batch_size, tf.float32)
        self.start_state = [(tf.placeholder(tf.float32, [args.batch_size, args.hidden_size]), \
                            tf.placeholder(tf.float32, [args.batch_size, args.hidden_size])) \
                            for _ in range(args.num_layers)]
        
        with tf.variable_scope('model'):
            softmax_w = tf.get_variable('softmax_w', [args.hidden_size, args.vocab_size])
            softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
            embedding = tf.get_variable('embedding', [args.vocab_size, args.hidden_size])

            embedded = tf.nn.embedding_lookup(embedding, self.input)
            inputs = tf.unpack(embedded, axis=1)

            state = self.start_state
            outputs = []
            states = []

            for i, inp in enumerate(inputs):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()

                output, state = self.core(inp, state)
                states.append(state)
                outputs.append(output)

        self.end_state = states[-1]
        output = tf.reshape(tf.concat(1, outputs), [-1, args.hidden_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [tf.reshape(self.target, [-1])],
                                                [tf.ones([args.batch_size * args.seq_length])],
                                                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.lr = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32), None)
        trainables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainables), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, trainables))
