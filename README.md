# char-rnn-tensorflow
Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Forked from sherjilozair [char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow).

- Added separate preprocessing code
- Added validation and test splitting
- Use state as tuple in tensorflow
- Implement sampling with temperature
- Add dropout like [Zaremba et al.](https://arxiv.org/abs/1409.2329)

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# Basic Usage
To preprocess a text file in utf-8, run `python preprocess.py --input_file INPUT_FILE --data_dir DATA_DIR`.

To train with default parameters, run `python train.py --data_dir DATA_DIR --save_dir SAVE_DIR`.

To continue training, run `python train.py --init_from SAVE_DIR`.

To sample from a checkpointed model, `python sample.py --init_from SAVE_DIR`.

# Roadmap
- Benchmark performance
