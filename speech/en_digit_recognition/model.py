#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2019/06/08 09:36:44
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

import librosa
import os
import tensorflow as tf
import numpy as np

np.random.seed(2019)

def read_files(paths, threshold=None):
    labels, features = [], []

    one_hot_array = np.eye(10)

    np.random.shuffle(paths)

    for i, path in enumerate(paths, 1):

        if i % 1e2 == 0:
            print(f"Loading data {i}")

        label, _, _ = os.path.basename(path).split("_")
        label = one_hot_array[int(label)]
        
        wave, sr = librosa.load(path, mono=True)
        mfcc = librosa.feature.mfcc(wave, sr)
        mfcc = np.pad(mfcc, ((0, 0), (0, 100-len(mfcc[0]))), mode="constant", constant_values=0)

        features.append(mfcc)
        labels.append(label)

        if threshold is not None and i == threshold: break

    return np.array(features), np.array(labels)


def mean_normalize(features):
    std_value = features.std()
    mean_value = features.mean()
    return (features - mean_value) / std_value


def batch_iter(features, labels):
    batch_size = 32
    indices = np.arange(len(features) // batch_size)
    np.random.shuffle(indices)

    for i in range(len(features) // batch_size):
        select_indices = indices[i: (i+1)*batch_size]
        yield features[select_indices], labels[select_indices]



class ASRCNN:

    def __init__(self, config, width, height, num_classes):
        self.config = config
        self.shape = (width, height)
        self.num_classes = num_classes
        
        self.build_model()


    def build_model(self):
        width, height = self.shape
        num_classes = self.num_classes

        self.input_x = tf.placeholder(tf.float32, [None, width, height], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        input_x = tf.transpose(self.input_x, [0, 2, 1])
        pooled_outputs = []

        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope(f"conv_maxpool_{filter_size}"):
                print(f"conv_maxpool_{filter_size}")

                conv = tf.layers.conv1d(input_x, self.config.num_filters, filter_size, activation=tf.nn.relu)
                pooled = tf.reduce_max(conv, reduction_indices=[1])
                pooled_outputs.append(pooled)

        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)    
        pooled_reshape = tf.reshape(tf.concat(pooled_outputs, 1), [-1, num_filters_total])

        fc = tf.layers.dense(pooled_reshape, self.config.hidden_dim, activation=tf.nn.relu, name="fc1")
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)

        self.logits = tf.layers.dense(fc, num_classes, name="fc2")
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)

        self.loss = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



class config:

    num_filters = 64
    filter_sizes = [50, 30, 20, 20, 20, 20]
    hidden_dim = 50
    
    learning_rate = 0.001
    keep_prob = 0.9
    num_epochs = 10
    print_per_batch = 10
    save_per_batch = 10


def get_paths():
    base_path = "./data/"
    train_paths = list(map(lambda filename: os.path.join(base_path+"recordings", filename), 
                        os.listdir(os.path.join(base_path, "recordings"))))

    train_count = int(len(train_paths)*0.8)
    np.random.shuffle(train_paths)
    train_paths, valid_paths = train_paths[:train_count], train_paths[train_count:]

    test_paths = list(map(lambda filename: os.path.join(base_path+"test", filename),
                        os.listdir(os.path.join(base_path, "test"))))

    return train_paths, valid_paths, test_paths



def train(train_paths, valid_paths, test_paths):
    print("loading data....")
    train_features, train_labels = read_files(train_paths)
    train_features = mean_normalize(train_features)
    valid_features, valid_labels = read_files(valid_paths)
    valid_features = mean_normalize(valid_features)
    test_features, test_labels = read_files(test_paths)
    test_features = mean_normalize(test_features)

    width, height = 20, 100
    classes = 10

    print("buildingc model....")
    model = ASRCNN(config, width, height, classes)
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join("asr_model", "model.ckpt")

    tensorboard_train_dir = "tensorboard/train"
    tensorboard_valid_dir = "tensorboard/valid"

    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)

    total_batch = 0
    for epoch in range(1, config.num_epochs+1):
        print(f"Epoch: {epoch}")

        for batch_x, batch_y in batch_iter(train_features, train_labels):
            total_batch += 1
            feed_dict = {
                model.input_x: batch_x,
                model.input_y: batch_y,
                model.keep_prob: config.keep_prob,
            }
            session.run(model.optimizer, feed_dict=feed_dict)

            if total_batch % config.print_per_batch == 0:
                train_loss, train_accuracy = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                valid_loss, valid_accuracy = session.run([model.loss, model.accuracy], feed_dict={
                    model.input_x: valid_features,
                    model.input_y: valid_labels,
                    model.keep_prob: config.keep_prob,
                })

                print(f"Steps: {total_batch}")
                print(f"train_loss: {train_loss},\t train_accuracy: {train_accuracy}")
                print(f"valid_loss: {valid_loss},\t valid_accuracy: {valid_accuracy}")
            

            if total_batch % config.save_per_batch == 0:
                train_summary = session.run(merged_summary, feed_dict=feed_dict)
                valid_summary = session.run(merged_summary, feed_dict={
                    model.input_x: valid_features,
                    model.input_y: valid_labels,
                    model.keep_prob: config.keep_prob,
                })

                train_writer.add_summary(train_summary, total_batch)
                valid_writer.add_summary(valid_summary, total_batch)

        saver.save(session, checkpoint_path, global_step=epoch)
        test_loss, test_accuracy = session.run([model.loss, model.accuracy], feed_dict={
            model.input_x: test_features,
            model.input_y: test_labels,
            model.keep_prob: config.keep_prob,
        })
        print(f"test_loss is {test_loss}, \t test_accuracy is {test_accuracy}\n\n")


if __name__ == "__main__":
    train_paths, valid_paths, test_paths = get_paths()

    train(train_paths, valid_paths, test_paths)

    

    
