# -*- coding: utf-8 -*-
import tensorflow as tf
import codecs
import os

class StandardInputFnConfig(object):

    def __init__(self, filenames, field_delim, clf_file, continuous_feature_number, discrete_feature_number,
            num_epochs=1, batch_size=128, buffer_size=256, perform_shuffle=False, drop_remainder=False):

        self.filenames = filenames
        self.field_delim = field_delim
        self.clf_file = clf_file
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.perform_shuffle = perform_shuffle
        self.drop_remainder = drop_remainder
        self.continuous_feature_number = continuous_feature_number
        self.discrete_feature_number = discrete_feature_number
        self.mult_feature_number = 1
        self.mult_feature_val_max_num = 50

        self.c_columns, self.d_columns, self.mult_columns = [], [], []
        with codecs.open(clf_file, 'r', 'UTF-8') as inf:
            for line in inf:
                line = line.strip()
                if len(line) < 1:
                    continue
                fs = line.split('\t')
                # cont    agentCate_ctr1days  1
                if fs[0] == 'cont' and len(fs) >= 3:
                    self.c_columns.append(str(fs[1]))
                elif fs[0] == 'cate' and len(fs) >= 4:
                    self.d_columns.append(str(fs[1]))
                elif fs[0] == 'var' and len(fs) >= 4:
                    self.mult_columns.append(str(fs[1]))
                elif fs[0] == 'mult' and len(fs) >= 4:
                    self.mult_columns.append(str(fs[1]))
        #self.c_columns = ['C' + str(i) for i in range(1, continuous_feature_number + 1)]
        #self.d_columns = ['D' + str(i) for i in range(continuous_feature_number+1, continuous_feature_number+discrete_feature_number+1)]

        self.c_columns_defaults = [[0.0] for i in range(continuous_feature_number)]
        self.d_columns_defaults = [[0] for i in range(discrete_feature_number)]
        self.mult_columns_defaults = [[""] for i in range(self.mult_feature_number)]
        self.feature_names = self.c_columns + self.d_columns
        self.feature_names_weight = ["weight"] + self.c_columns + self.d_columns
        self.feature_names_mult = self.c_columns + self.d_columns + self.mult_columns
        self.columns_defaults = [[0]] + self.c_columns_defaults + self.d_columns_defaults
        self.columns_weight_defaults = [[1.0], [0]] + self.c_columns_defaults + self.d_columns_defaults
        self.columns_mult_defaults = [[0]] + self.c_columns_defaults + self.d_columns_defaults + self.mult_columns_defaults


    def parse_csv(self, line):
        columns = tf.decode_csv(line, self.columns_defaults, field_delim=self.field_delim)
        #label = columns[:1]
        label = columns[0]
        del columns[0]
        features = dict(zip(self.feature_names, columns))
        return features, label

    def parse_csv_weight(self, line):
        columns = tf.decode_csv(line, self.columns_weight_defaults, field_delim=self.field_delim)
        # weight label fea1 fea2 ...
        label = columns[1]
        del columns[1]
        features = dict(zip(self.feature_names_weight, columns))
        return features, label

    def parse_csv_mult(self, line):
        columns = tf.decode_csv(line, self.columns_mult_defaults, field_delim=self.field_delim)
        #label = columns[:1]
        label = columns[0]
        del columns[0]
        features = dict(zip(self.feature_names_mult, columns))
        for col in self.mult_columns:
            mult_val = features[col]
            kvpairs = tf.string_split([mult_val], delimiter=",").values[:self.mult_feature_val_max_num]
            kvpairs = tf.string_split(kvpairs, ':').values
            kvpairs = tf.reshape(kvpairs, [-1, 2])
            #feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)
            feat_ids, feat_vals = kvpairs[:,0], kvpairs[:,1]
            feat_ids= tf.strings.to_number(feat_ids, out_type=tf.int32)
            feat_vals= tf.strings.to_number(feat_vals, out_type=tf.float32) #N*1
            features[col] = feat_ids
            features[col+"_weight"] = feat_vals
        return features, label


    def input_fn(self):
        print('Parsing:', self.filenames)
        dataset = tf.data.TextLineDataset(self.filenames)
        dataset = dataset.map(self.parse_csv, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.perform_shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=2021)
        dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def input_weight_fn(self):
        print('Parsing Weight File:', self.filenames)
        dataset = tf.data.TextLineDataset(self.filenames)
        dataset = dataset.map(self.parse_csv_weight, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.perform_shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=2021)
        dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def input_mult_fn(self):
        filename_list = []
        if os.path.isdir(self.filenames):
            filename_list = os.listdir(self.filenames)
            filename_list = [self.filenames+"/"+f for f in filename_list]
        else:
            filename_list = [self.filenames]
        print('Parsing:', self.filenames, filename_list)
        dataset = tf.data.TextLineDataset(filename_list)
        dataset = dataset.map(self.parse_csv_mult, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.perform_shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=2021)
        dataset = dataset.repeat(self.num_epochs)

        pad_shapes = {}
        pad_values = {}
        for col in self.c_columns:
            pad_shapes[col] = tf.TensorShape([])
            pad_values[col] = 0.
        for col in self.d_columns:
            pad_shapes[col] = tf.TensorShape([])
            pad_values[col] = 0
        for col in self.mult_columns:
            pad_shapes[col] = tf.TensorShape([self.mult_feature_val_max_num])
            pad_values[col] = 0
            pad_shapes[col+"_weight"] = tf.TensorShape([self.mult_feature_val_max_num])
            pad_values[col+"_weight"] = tf.constant(0.0, dtype=tf.float32)
        pad_shapes = (pad_shapes, (tf.TensorShape([])))
        pad_values = (pad_values, (tf.constant(0, dtype=tf.int32)))

        dataset = dataset.padded_batch(batch_size=self.batch_size, padded_shapes=pad_shapes, padding_values=pad_values, drop_remainder=self.drop_remainder)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

def load_feature_infos(feature_classify_file):
    #import logging
    cate_feats = list()
    cate_feat_val_num = list()
    dense_feats = list()
    mult_feats = list()
    mult_feat_val_num = list()
    with codecs.open(feature_classify_file, 'r', 'UTF-8') as inf:
        for line in inf:
            line = line.strip()
            if len(line) < 1:
                continue
            fs = line.split('\t')
            if fs[0] == 'cont' and len(fs) >= 3:
                dense_feats.append(str(fs[1]))
            elif fs[0] == 'cate' and len(fs) >= 4:
                # cate    creative    67  1296
                cate_feats.append(str(fs[1]))
                cate_feat_val_num.append(int(fs[3]))
            elif fs[0] == 'mult' and len(fs) >= 4:
                mult_feats.append(str(fs[1]))
                mult_feat_val_num.append(int(fs[3]))
            elif fs[0] == 'var' and len(fs) >= 4:
                mult_feats.append(str(fs[1]))
                mult_feat_val_num.append(int(fs[3]))
    print('cate_feats size: {}, cate_feat_val_num size: {}, dense_feats size: {}, mult_feats size: {}'.format(
        len(cate_feats), len(cate_feat_val_num), len(dense_feats), len(mult_feats)
    ))
    return {'cate_feats':cate_feats, 'cate_feat_val_num':cate_feat_val_num, 'dense_feats':dense_feats, 'mult_feats':mult_feats, 'mult_feat_val_num':mult_feat_val_num}

