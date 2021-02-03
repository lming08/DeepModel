# -*- coding: utf-8 -*-

"""

@file: naive: read data from feather format file, variable multiple feature average
@user: apple.py
@time: 2020-03-03 11:18
@author: sxd

"""

import sys
import os
import logging
import feather
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.saved_model.signature_def_utils import predict_signature_def

class NaiveChunkLoad(object):

    def __init__(self, hparams):
        """
        self.init_method
        self.init_value
        self.opt_method
        self.learning_rate
        self.reg_method
        self.l1_scale
        self.l2_scale
        self.activate_method
        self.loss_method
        self.metrics_name
        self.batch_norm
        self.drop_out
        self.cate_feats
        self.cate_feat_val_num
        self.dense_feats
        self.is_use_dense
        self.mult_feats
        self.mult_val_num
        self.com_feats_name
        self.layer_dim_lst
        self.labels
        self.is_norm
        self.decay_method_id
        self.decay_steps
        self.decay_rate
        self.near_zero
        self.checkpoint_dir
        self.calibration_ratio
        self.is_dense_auto_project

        self.all_emb_dim
        self.cate_input_lst
        self.labels
        self.initializer
        self.activation
        self.regularizer
        self.keep_prob
        self.deep_output
        self.global_step
        self.loss
        self.opt
        :param hparams:
        """

        tf.set_random_seed(1234)
        self.init_method = hparams['init_method']
        self.init_value = hparams['init_value']
        self.opt_method = hparams['opt_method']
        self.learning_rate = hparams['learning_rate']
        self.reg_method = hparams['reg_method']
        self.l1_scale = hparams['l1_scale']
        self.l2_scale = hparams['l2_scale']
        self.activate_method = hparams['activate_method']
        self.loss_method = hparams['loss_method']
        self.metrics_name = hparams['metrics']
        self.batch_norm = hparams['batch_norm']
        self.drop_out = hparams['drop_out']
        self.cate_feats = hparams['cate_feats']
        self.cate_feat_val_num = hparams['cate_feat_val_num']
        self.dense_feats = hparams['dense_feats']
        self.mult_feats = hparams['mult_feats']
        self.mult_val_num = hparams['mult_val_num']
        self.com_feats_name = hparams['com_feats_name']
        self.is_use_dense = hparams['is_use_dense']
        self.layer_dim_lst = hparams['layer_dim_lst']
        self.label_name = hparams['label_name']
        self.is_norm = hparams['is_norm']
        self.decay_method_id = hparams['decay_method_id']
        self.decay_steps = hparams['decay_steps']
        self.decay_rate = hparams['decay_rate']
        self.near_zero = hparams['near_zero']
        self.checkpoint_dir = hparams['checkpoint_dir']
        self.calibration_ratio = hparams['calibration_ratio']
        self.is_dense_auto_project = hparams['is_dense_auto_project']

        self.metrics = self._get_metrics(self.metrics_name)
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        self._init_some_params()
        self._build_model()
        self._init_evn()

    def _init_some_params(self):
        self.initializer = self._get_init_method(self.init_method, self.init_value)
        self.activation = self._get_activate_method(self.activate_method)
        self.regularizer = self._get_regularizer(self.reg_method, self.l1_scale, self.l2_scale)

    def _get_init_method(self, init_method, init_value):
        initializer = tf.truncated_normal_initializer(stddev=init_value)
        if init_method == 'tnormal':
            initializer = tf.truncated_normal_initializer(stddev=init_value)
        elif init_method == 'uniform':
            initializer = tf.random_uniform_initializer(-init_value, init_value)
        elif init_method == 'normal':
            initializer = tf.random_normal_initializer(stddev=init_value)
        elif init_method == 'xavier_normal':
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        elif init_method == 'xavier_uniform':
            initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        elif init_method == 'he_normal':
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
        elif init_method == 'he_uniform':
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
        return initializer

    def _get_opt_method(self, opt_method, learning_rate):
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        if opt_method == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate)
        elif opt_method == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif opt_method == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif opt_method == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif opt_method == 'ftrl':
            opt = tf.train.FtrlOptimizer(learning_rate)
        elif opt_method == 'gd':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif opt_method == 'padagrad':
            opt = tf.train.ProximalAdagradOptimizer(learning_rate)
        elif opt_method == 'pgd':
            opt = tf.train.ProximalGradientDescentOptimizer(learning_rate)
        elif opt_method == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate)
        return opt

    def _get_regularizer(self, reg_method, l1_scale, l2_scale):
        regularizer = tf.contrib.layers.l2_regularizer(l2_scale)
        if reg_method == 'l1':
            regularizer = tf.contrib.layers.l1_regularizer(l1_scale)
        elif reg_method == 'l2':
            regularizer = tf.contrib.layers.l2_regularizer(l2_scale)
        elif reg_method == 'l1_l2':
            regularizer = tf.contrib.layers.l1_l2_regularizer(l1_scale, l2_scale)
        return regularizer

    def _get_learning_rate_decay(self, decay_method_id, init_lr, global_step, decay_steps, decay_rate):
        learning_rate_decayer = None
        if decay_method_id == 1:
            learning_rate_decayer = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step,
                                                               decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
        elif decay_method_id == 2:
            learning_rate_decayer = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step,
                                                               decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
        elif decay_method_id == 3:
            learning_rate_decayer = tf.train.polynomial_decay(learning_rate=init_lr, global_step=global_step,
                                                              decay_steps=decay_steps, end_learning_rate=0.0000001,
                                                              power=1, cycle=False)
        elif decay_method_id == 4:
            learning_rate_decayer = tf.train.polynomial_decay(learning_rate=init_lr, global_step=global_step,
                                                              decay_steps=decay_steps, end_learning_rate=0.0000001,
                                                              power=1, cycle=True)
        elif decay_method_id == 5:
            learning_rate_decayer = tf.train.polynomial_decay(learning_rate=init_lr, global_step=global_step,
                                                              decay_steps=decay_steps, end_learning_rate=0.0000001,
                                                              power=2, cycle=False)
        elif decay_method_id == 6:
            learning_rate_decayer = tf.train.polynomial_decay(learning_rate=init_lr, global_step=global_step,
                                                              decay_steps=decay_steps, end_learning_rate=0.0000001,
                                                              power=2, cycle=True)
        elif decay_method_id == 7:
            learning_rate_decayer = tf.train.polynomial_decay(learning_rate=init_lr, global_step=global_step,
                                                              decay_steps=decay_steps, end_learning_rate=0.0000001,
                                                              power=0.5, cycle=False)
        elif decay_method_id == 8:
            learning_rate_decayer = tf.train.polynomial_decay(learning_rate=init_lr, global_step=global_step,
                                                              decay_steps=decay_steps, end_learning_rate=0.0000001,
                                                              power=0.5, cycle=True)
        elif decay_method_id == 9:
            learning_rate_decayer = tf.train.inverse_time_decay(learning_rate=init_lr, global_step=global_step,
                                                                decay_steps=decay_steps, decay_rate=decay_rate,
                                                                staircase=False)
        elif decay_method_id == 10:
            learning_rate_decayer = tf.train.inverse_time_decay(learning_rate=init_lr, global_step=global_step,
                                                                decay_steps=decay_steps, decay_rate=decay_rate,
                                                                staircase=True)
        elif decay_method_id == 11:
            learning_rate_decayer = tf.train.cosine_decay(learning_rate=init_lr, global_step=global_step,
                                                          decay_steps=decay_steps)
        elif decay_method_id == 12:
            learning_rate_decayer = tf.train.cosine_decay_restarts(learning_rate=init_lr, global_step=global_step,
                                                                   first_decay_steps=decay_steps)
        elif decay_method_id == 13:
            learning_rate_decayer = tf.train.linear_cosine_decay(learning_rate=init_lr, global_step=global_step,
                                                                 decay_steps=decay_steps)
        elif decay_method_id == 14:
            learning_rate_decayer = tf.train.noisy_linear_cosine_decay(learning_rate=init_lr, global_step=global_step,
                                                                       decay_steps=decay_steps)
        return learning_rate_decayer

    def _get_activate_method(self, activate_method):
        activate = None
        if activate_method == 'sigmoid':
            activate = tf.nn.sigmoid
        elif activate_method == 'softmax':
            activate = tf.nn.softmax
        elif activate_method == 'relu':
            activate = tf.nn.relu
        elif activate_method == 'tanh':
            activate = tf.nn.tanh
        elif activate_method == 'elu':
            activate = tf.nn.elu
        elif activate_method == 'identity':
            activate = tf.identity
        else:
            raise ValueError("this activations not defined {0}".format(activate_method))
        return activate

    def _get_metrics(self, metrics_name):
        metrics = None
        if metrics_name == 'auc':
            metrics = roc_auc_score
        else:
            raise ValueError("this metrics not defined {0}".format(metrics_name))
        return metrics

    def dense_auto_project_layer(self, inputs, split_num, split_axis, unit, initializer, is_batch_normal, name='auto_project'):
        splits = tf.split(inputs, split_num, axis=split_axis)
        projects = list()
        for sp in splits:
            if is_batch_normal:
                sp = tf.layers.batch_normalization(sp)
            project = tf.layers.dense(sp, unit, activation=tf.identity, kernel_initializer=initializer, bias_initializer=initializer)
            projects.append(project)
        auto_project = tf.concat(projects, axis=1, name=name)
        return auto_project

    def _build_input(self, cate_feats, cate_feat_val_num, dense_feats, mult_feats, com_feats, initializer):
        # cate features
        self.cate_input_lst = list()
        col_emb_lst = list()
        self.all_emb_dim = 0
        for index, col in enumerate(cate_feats):
            val_num = cate_feat_val_num[index]
            col_input = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='cate_input_' + col)
            self.cate_input_lst.append(col_input)
            col_emb_dim = self._get_emb_dim(val_num)
            self.all_emb_dim += col_emb_dim
            col_weight = tf.get_variable(dtype=tf.float32, shape=[val_num + 1, col_emb_dim],
                                         initializer=initializer, name='cate_weight_' + col)
            # col_input 该维度onehot序号
            col_emb = tf.nn.embedding_lookup(col_weight, col_input, max_norm=1, name='cate_emb_' + col)
            col_emb_lst.append(col_emb)

        # mult features
        # mult_weight = tf.get_variable(dtype=tf.float32, shape=[self.mult_val_num + 1, 32],
        #                               initializer=initializer, name='mult_weight')
        # self.mult_input = tf.placeholder(dtype=tf.int32, shape=[None, len(self.mult_feats)], name='mult_input')
        # mult_input_mask_b = tf.greater(self.mult_input, 0, name='mult_input_mask_b')
        # mult_input_mask_f = tf.cast(mult_input_mask_b, dtype=tf.float32, name='mult_input_mast_f')
        # mult_input_num = tf.reduce_sum(mult_input_mask_f, axis=1, keep_dims=True) + self.near_zero
        # mult_input_mask_exp = tf.expand_dims(mult_input_mask_f, axis=2, name='mult_input_mask_exp')
        # mult_emb = tf.nn.embedding_lookup(mult_weight, self.mult_input, max_norm=1, name='mult_emb')
        # mult_emb_mask = tf.multiply(mult_emb, mult_input_mask_exp, name='mult_emb_mask')
        # mult_emb_sum = tf.reduce_sum(mult_emb_mask, axis=1, name='mult_emb_sum')
        # mult_emb_avg = tf.div(mult_emb_sum, mult_input_num, name='mult_emb_avg')
        # mult_emb_avg_exp = tf.expand_dims(mult_emb_avg, axis=1, name='mult_emb_avg_exp')
        # com features
        # self.com_input = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='com_input_' + com_feats)
        # com_emb = tf.nn.embedding_lookup(mult_weight, self.com_input, max_norm=1, name='com_emb_' + com_feats)
        # emb_input = tf.concat(col_emb_lst + [mult_emb_avg_exp, com_emb], axis=2)
        emb_input = tf.concat(col_emb_lst, axis=2) #指定axis表示该维度发生改变
        emb_input = tf.layers.flatten(emb_input, name='emb_input')
        # dense features
        if self.is_use_dense:
            self.dense_input = tf.placeholder(dtype=tf.float32, shape=[None, len(dense_feats)], name='dense_input')
            if self.is_dense_auto_project:
                auto_project_dense_input = self.dense_auto_project_layer(self.dense_input, split_num=len(dense_feats), split_axis=1,
                                                                         unit=1, initializer=self.initializer,
                                                                         is_batch_normal=self.batch_norm, name='dense_auto_project_input')
                auto_project_input = tf.concat([emb_input, auto_project_dense_input], axis=1, name='deep_auto_project_input_0')
                return auto_project_input
            input = tf.concat([emb_input, self.dense_input], axis=1, name='deep_input_0')
            return input
        return emb_input

    def _build_deep(self, input, layer_dim_lst, activation, initializer, regularizer, keep_prob):
        deep_output = input
        for i, dim in enumerate(layer_dim_lst):
            if self.batch_norm:
                deep_output = tf.layers.batch_normalization(deep_output)
            deep_output = tf.layers.dense(inputs=deep_output, units=dim, activation=activation,
                                          kernel_initializer=initializer, bias_initializer=initializer,
                                          kernel_regularizer=regularizer, name='deep_input_' + str(i + 1))
            deep_output = tf.layers.dropout(deep_output, keep_prob)
        return deep_output

    def _build_model(self):
        # self.initializer = self._get_init_method(self.init_method, self.init_value)
        # self.activation = self._get_activate_method(self.activate_method)
        # self.regularizer = self._get_regularizer(self.reg_method, self.l1_scale, self.l2_scale)
        # self.keep_prob = tf.placeholder(tf.float32, name='droup_out')
        self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
        deep_input = self._build_input(self.cate_feats, self.cate_feat_val_num, self.dense_feats, self.mult_feats,
                                       self.com_feats_name, self.initializer)
        deep_output = self._build_deep(deep_input, self.layer_dim_lst,
                                       self.activation, self.initializer, self.regularizer, self.drop_out)
        deep_output = tf.layers.dense(deep_output, 1, activation=tf.nn.sigmoid,
                                           kernel_initializer=self.initializer, bias_initializer=self.initializer,
                                           kernel_regularizer=self.regularizer, name='out_sigmoid')
        self.deep_output = tf.identity(deep_output, name='output')
        self.global_step = tf.Variable(0, trainable=False)
        self.loss = tf.losses.log_loss(self.labels, self.deep_output)
        # regularizer_loss = tf.losses.get_regularization_loss()
        # self.loss += regularizer_loss
        self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=self.global_step,
                                                        decay_steps=self.decay_steps, decay_rate=0.99)
        self.opt = self._get_opt_method(self.opt_method, self.learning_rate).minimize(self.loss, self.global_step)

    def _init_evn(self):
        # GPU显存按需分配
        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.gpu_config)

    def _get_emb_dim(self, val_num):
        log_val = np.log2(val_num + 1)
        dim = int(log_val)
        if log_val - dim > 0.1:
            dim += 1
        return dim

    def train(self, train_file, valid_file, batch_size, block_num, epochs, verb, model_dir):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        columns = [self.label_name] + self.dense_feats + self.cate_feats #+ self.mult_feats + [self.com_feats_name]
        usecols = range(0, len(columns))
        #df_train = self._load_data(train_file, columns, usecols)
        df_valid = self._load_data(valid_file, columns, usecols)
        for epoch in range(0, epochs):
            logging.info('-----start training epoch {}-----'.format(epoch))
            '''
            batchs = self._get_batch(df_train, batch_size, is_sample=True, is_norm=self.is_norm)
            loss_sum = 0
            for df_batch in batchs:
                loss, step = self._train_batch(df_batch, self.sess)
                loss_sum += loss
                if verb > 0 and step % verb == 0:
                    logging.info('epoch [{}]\t{} steps\tloss={}'.format(epoch, step, loss_sum / verb))
                    loss_sum = 0
            '''
            loss_sum = 0
            for sub_data in pd.read_csv(train_file, sep='\t', names=columns, usecols=usecols, chunksize=batch_size*10000):
                batchs = self._get_batch(sub_data, batch_size, is_sample=True, is_norm=self.is_norm)
                for df_batch in batchs:
                    loss, step = self._train_batch(df_batch, self.sess)
                    loss_sum += loss
                    if verb > 0 and step % verb == 0:
                        logging.info('epoch [{}]\t{} steps\tloss={}'.format(epoch, step, loss_sum / verb))
                        loss_sum = 0
            #train_labels, train_preds = self._predict(df_train, batch_size, block_num, None, 'train', 0)
            #train_mval = self._evaluate(train_labels, train_preds, self.metrics)
            #train_logloss = self._evaluate_logloss(train_labels, train_preds, False)
            valid_labels, valid_preds = self._predict(df_valid, batch_size, block_num, None, 'valid', 0)
            valid_mval = self._evaluate(valid_labels, valid_preds, self.metrics)
            valid_logloss = self._evaluate_logloss(valid_labels, valid_preds, True)
            #logging.info('epoch [{}]\ttrain-{}: {}\ttrain-logloss: {}\tvalid-{}: {}\tvalid-logloss: {}'.format(
            #    epoch, self.metrics_name, train_mval, train_logloss,
            #           self.metrics_name, valid_mval, valid_logloss
            #))
            logging.info('epoch [{}]\tvalid-{}: {}\tvalid-logloss: {}'.format(
                epoch, self.metrics_name, valid_mval, valid_logloss
            ))
            #df_result = pd.DataFrame({'train_preds': train_preds})
            #logging.info('epoch [{}]\ttrain predict infos:\n{}'.format(epoch, df_result.describe()))
            df_result = pd.DataFrame({'valid_preds': valid_preds})
            logging.info('epoch [{}]\tvalid predict infos:\n{}'.format(epoch, df_result.describe()))
            if epoch == 1:
                logging.info('saving model to file')
                self.save(model_dir)

    def _train_batch(self, df_train, sess):
        feed_dict = dict()
        for index, col in enumerate(self.cate_feats):
            feed_dict[self.cate_input_lst[index]] = df_train[[col]].values
        # feed_dict[self.mult_input] = df_train[self.mult_feats].values
            # print df_train[[mult]].values
        if self.is_use_dense > 0:
            feed_dict[self.dense_input] = df_train[self.dense_feats].values
        # feed_dict[self.com_input] = df_train[[self.com_feats_name]].values
        feed_dict[self.labels] = df_train[[self.label_name]].values
        # feed_dict[self.keep_prob] = self.drop_out
        loss, _, step = sess.run([self.loss, self.opt, self.global_step], feed_dict=feed_dict)
        # global_step 随着每一个batch自增计数,表示迭代更新权重次数
        # print feed_dict[self.mult_input]
        # print exp
        # print(avg.shape)
        # print(avg[0].tolist())
        return loss, step

    def _load_data(self, data_file, columns, usecols):
        fth_file = data_file + '.fth'
        if not os.path.exists(fth_file):
            logging.info('convert csv file to feather')
            df_tmp = pd.read_csv(data_file, sep='\t', names=columns, usecols=usecols)
            logging.info('csv data shape {}'.format(df_tmp.shape))
            feather.write_dataframe(df_tmp, fth_file)
            df_tmp.head()
        logging.info('loading data {}'.format(fth_file))
        df_data = feather.read_dataframe(fth_file, columns=columns, use_threads=True)
        logging.info('data shape {}'.format(df_data.shape))
        #print df_data.head()['package_name']
        return df_data

    def _predict(self, df, batch_size, block_num, metrics=None, pred_type='test', verb=100):
        if pred_type not in ['test', 'train', 'valid']:
            raise ValueError("pred_type not in ['test', 'train', 'valid']")
        if pred_type == 'test':
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
        preds = list()
        labels = list()
        num = 0
        batchs = self._get_batch(df, batch_size, is_norm=self.is_norm)
        for df_batch in batchs:
            tmp_preds = self._predict_batch(df_batch, self.sess)
            preds += tmp_preds
            labels += df_batch[self.label_name].values.tolist()
            num += 1
            if verb > 0 and num % verb == 0:
                logging.info('already predicted {} data'.format(verb * batch_size))

        preds = np.concatenate(preds).reshape(-1,)
        if metrics:
            mval = self._evaluate(labels, preds, metrics)
            logging.info('{}: {}'.format(self.metrics_name, mval))

        return labels, preds

    # def predict(self, data_file, batch_size, block_num, metrics=None, pred_type='test', verb=100):
    #     if pred_type not in ['test', 'train', 'valid']:
    #         raise ValueError("pred_type not in ['test', 'train', 'valid']")
    #     columns = [self.label_name] + self.dense_feats + self.cate_feats + self.mult_feats + [self.com_feats_name]
    #     usecols = range(0, len(columns))
    #     if pred_type == 'test':
    #         self.sess.run(tf.global_variables_initializer())
    #         self.sess.run(tf.local_variables_initializer())
    #         columns = self.dense_feats + self.cate_feats
    #         block_num = 1
    #     iter = pd.read_csv(data_file, sep='\t', usecols=usecols, names=columns, iterator=True, chunksize=batch_size * block_num)
    #     preds = list()
    #     labels = list()
    #     num = 0
    #     while True:
    #         try:
    #             df = iter.get_chunk()
    #             batchs = self._get_batch(df, batch_size * 10, is_norm=self.is_norm)
    #             for df_batch in batchs:
    #                 tmp_preds = self._predict_batch(df_batch, self.sess)
    #                 preds += tmp_preds
    #                 labels += df_batch[self.label_name].values.tolist()
    #                 num += 1
    #                 if verb > 0 and num % verb == 0:
    #                     logging.info('already predicted {} data'.format(verb * batch_size))
    #         except StopIteration:
    #             break
    #     preds = np.concatenate(preds).reshape(-1,)
    #     if metrics:
    #         mval = self._evaluate(labels, preds, metrics)
    #         logging.info('{}: {}'.format(self.metrics_name, mval))
    #
    #     return labels, preds

    def _predict_batch(self, df_data, sess):
        feed_dict = dict()
        for index, col in enumerate(self.cate_feats):
            feed_dict[self.cate_input_lst[index]] = df_data[[col]].values
        # feed_dict[self.mult_input] = df_data[self.mult_feats].values
        if self.is_use_dense > 0:
            feed_dict[self.dense_input] = df_data[self.dense_feats].values
        # feed_dict[self.com_input] = df_data[[self.com_feats_name]].values
        #feed_dict[self.keep_prob] = self.drop_out
        preds = sess.run([self.deep_output], feed_dict=feed_dict)
        return preds

    def _evaluate(self, labels, preds, metrics):
        mval = metrics(labels, preds)
        mval = round(mval, 6)
        return mval

    def _evaluate_logloss(self, labels, preds, is_calibration):
        if is_calibration:
            preds = self._calibration(preds, self.calibration_ratio)
        mval = log_loss(labels, preds)
        mval = round(mval, 6)
        return mval

    def _calibration(self, preds, ratio):
        return preds * ratio / (preds * (ratio - 1) + 1)

    def _get_batch(self, df, batch_size, is_sample=False, is_norm=False):
        batchs = list()
        if is_sample:
            df = df.sample(df.shape[0], random_state=1234)
        if is_norm:
            for col in self.dense_feats:
                max_val = df[col].max()
                min_val = df[col].min()
                scale = max_val - min_val
                if scale == 0:
                    scale = 1
                df[col] = (df[col] - min_val) / scale
        i = 0
        while True:
            bg = i * batch_size
            if bg >= df.shape[0]:
                break
            ed = bg + batch_size
            df_batch = df.iloc[bg:ed]
            batchs.append(df_batch)
            i += 1

        return batchs

    def save(self, model_dir):
        # builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
        # input_dic = {
        #     'cate_input_lst': self.cate_input_lst,
        #     'mult_input': self.mult_input,
        #     'com_input': self.com_input,
        #     'dense_input': self.dense_input
        # }
        #
        # signature = predict_signature_def(inputs=input_dic,
        #                                   outputs={'score': self.deep_output})
        # builder.add_meta_graph_and_variables(self.sess, tags=['mytag'], signature_def_map={'predict': signature})
        # builder.save(as_text=True)
        saver = tf.train.Saver()
        saver.save(self.sess, model_dir, self.global_step)




