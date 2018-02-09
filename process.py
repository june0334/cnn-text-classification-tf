#!/usr/bin/env python
#coding=utf8

import json
import os
import numpy as np
import nltk


def load_data(ffile, out_dir, test_sample_percentage, shuffle=True):
    if os.path.exists(ffile) and os.path.isfile(ffile):
        data = json.load(open(ffile, 'r'))
        intents = ["greeting", "intent_resturant_search", "slots_wait", "slots_fill", "confirm"]
        new_data = [(d["utterances"][-1], d["intent"]) for d in data if d["utterances"][-1] != "<silence>" and d["intent"] and d["intent"] in intents]
        utterance_data, intent_data = zip(*new_data)
        data_size = len(intent_data)
        if shuffle:
            np.random.shuffle(utterance_data)
            np.random.shuffle(intent_data)
        test_sample_index = -1 * int(test_sample_percentage * float(data_size))
        utterance_train, utterance_test = utterance_data[:test_sample_index], utterance_data[test_sample_index:]
        intent_train, intent_test = intent_data[:test_sample_index], intent_data[test_sample_index:]
        data_dir = os.path.join(out_dir, "model_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(os.path.join(data_dir, "train"), "w") as train_file:
            map(lambda x: train_file.write("||".join(x) + "\n"), list(zip(utterance_train, intent_train)))
        with open(os.path.join(data_dir, "test"), "w") as test_file:
            map(lambda x: test_file.write("||".join(x) + "\n"), list(zip(utterance_test, intent_test)))
        utterance_train = [nltk.word_tokenize(instance) for instance in utterance_train]
        return utterance_train, intent_train


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            np.random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]
