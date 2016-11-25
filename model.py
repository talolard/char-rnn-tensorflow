from __future__ import division
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, array_ops
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import variable_scope


class Model():

    def __init__(self, args):
        layer_type = rnn_cell.BasicLSTMCell
        layer = layer_type(args.hidden_size, state_is_tuple=False)
        self.dropout = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32), None)
        wrapped = tf.nn.rnn_cell.DropoutWrapper(layer, input_keep_prob=self.dropout)
        self.core = rnn_cell.MultiRNNCell([wrapped] * args.num_layers, state_is_tuple=False)

        self.input = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.target = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        self.initial_state_cell = self.core.zero_state(args.batch_size, tf.float32)
        self.initial_state_attn = array_ops.zeros([args.batch_size, args.seq_length, args.hidden_size])

        self.cell_state_placeholder =tf.placeholder(dtype=tf.float32,name="cell_state",shape=[args.batch_size,2*args.hidden_size])
        self.attn_state_placeholder = tf.placeholder(dtype=tf.float32, name="attn_state",shape=[args.batch_size, args.seq_length, args.hidden_size])


        with tf.variable_scope('model'):
            cell_state = self.cell_state_placeholder
            attn_state = self.attn_state_placeholder
            softmax_w = tf.get_variable('softmax_w', [args.hidden_size, args.vocab_size])
            softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
            embedding = tf.get_variable('embedding', [args.vocab_size, args.hidden_size])

            embedded = tf.nn.embedding_lookup(embedding, self.input)
            inputs = tf.unpack(embedded, axis=1)

            outputs = []
            states = []


            outputs, state = seq2seq.attention_decoder(decoder_inputs=inputs,
                                                        initial_state=self.initial_state_cell,
                                                        attention_states=self.initial_state_attn,
                                                        cell=self.core,
                                                       initial_state_attention=True

                                                        )

        self.end_state = state
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
