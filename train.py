#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import process
from tqdm import tqdm
from text_cnn import TextCNN
from tensorflow.contrib import learn

import pdb

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("intention_data_file", "/nfs/project/data/dataset-E2E-goal-oriented/dialog-task1API-kb1_atmosphere-distr0.5-trn10000-new.json", "Data source for intention classification")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("glove_dir", "/nfs/project/data/glove/", "glove data directory")
tf.flags.DEFINE_string("glove_corpus", "6B", "glove corpus (default: 6B)")
tf.flags.DEFINE_integer("glove_vec_size", 100, "glove vector size (default: 100)")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_string("embedding_style", "glove", "embedding style (default: use glove vector)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
x, y = process.load_data(FLAGS.intention_data_file, out_dir, float(0.2), True)

# corpus words
vocabulary = set()
for sentence in x:
    for word in sentence:
        vocabulary.add(word)

vocab_list = list(vocabulary)
# word index
vocab_dict = {}
for i in range(len(vocab_list)):
    vocab_dict[vocab_list[i]] = i+1
print('dict size:' + str(len(vocab_dict)))

x = list(map(lambda sentence: [vocab_dict[word] for word in sentence], x))
max_sentence_length = max(list(map(len, x)))
'''
x = np.array(list(map(lambda sentence: sentence.append([0] * (max_sentence_length - len(sentence))) if len(sentence) <
                                                                               max_sentence_length else sentence, x)))
'''
for sentence in x:
    if len(sentence) < max_sentence_length:
       sentence[len(sentence):len(sentence)] = [0] * (max_sentence_length - len(sentence))

# label list
intents = ["greeting", "intent_resturant_search", "slots_wait", "slots_fill", "confirm"]
intents_dict = {}
for i in range(len(intents)):
    intents_dict[intents[i]] = i
tmp = [0.0] * len(intents)
# sample label
extend_y = []
for intent in y:
    curr = tmp[:]
    curr[intents_dict[intent]] = 1.0 
    extend_y.append(curr)
y = extend_y

# map word id to vector
glove_data_path = os.path.join(FLAGS.glove_dir, "glove.{}.{}d.txt".format(FLAGS.glove_corpus, FLAGS.glove_vec_size))
sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
total = sizes[FLAGS.glove_corpus]
id_vector_dict = {}
with open(glove_data_path, 'r', encoding='utf8') as gd:
    for line in tqdm(gd, total=total):
        array = line.strip().split(" ")
        word = array[0]
        vector = list(map(float, array[1:]))
        if word in vocab_dict:
            id_vector_dict[vocab_dict[word]] = vector
        elif word.capitalize() in vocab_dict:
            id_vector_dict[vocab_dict[word.capitalize()]] = vector
        elif word.lower() in vocab_dict:
            id_vector_dict[vocab_dict[word.lower()]] = vector
        elif word.upper() in vocab_dict:
            id_vector_dict[vocab_dict[word.upper()]] = vector
print("{}/{} of word vocab have corresponding vectors in {}".format(len(id_vector_dict), len(vocab_dict), glove_data_path))

'''
for sequence in x:
    for index in range(len(sequence)):
        if sequence[index] in id_vector_dict:
            sequence[index] = id_vector_dict[sequence[index]]
        else:
            sequence[index] = [0] * FLAGS.glove_vec_size
'''
x = list(map(lambda sentence: list(map(lambda word: id_vector_dict[word] if word in id_vector_dict else [float(0)] * FLAGS.glove_vec_size, sentence)), x))
pdb.set_trace()
#id_vector_dict = np.array([id_vector_dict[id + 1] if (id + 1) in id_vector_dict else
#                           np.zeros(FLAGS.glove_vec_size) for id in range(len(vocab_list))])


# Build vocabulary
#max_document_length = max([len(x.split(" ")) for x in x_text])
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
#np.random.seed(10)
#shuffle_indices = np.random.permutation(np.arange(len(y)))
#x_shuffled = x[shuffle_indices]
#y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

#del x, y, x_shuffled, y_shuffled

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
# Trainingimport pdb
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=max_sentence_length,
            num_classes=len(intents),
            vocabulary=vocab_dict,
            glove_vacabulary=id_vector_dict,
            glove_embedding_size=FLAGS.glove_vec_size,
            embedding_style=FLAGS.embedding_style,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = process.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
