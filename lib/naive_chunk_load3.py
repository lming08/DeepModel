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
import threading

class NaiveChunkLoad3(object):

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
        elif opt_method == 'sgdm':
            opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
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
                sp = tf.layers.batch_normalization(sp, momentum = 0.9, training=self.training)
            project = tf.layers.dense(sp, unit, use_bias=True, activation=tf.identity, kernel_initializer=initializer, bias_initializer=initializer)
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

    def _build_deep(self, input, layer_dim_lst, activation, initializer, regularizer, dropout_prob):
        deep_output = input
        for i, dim in enumerate(layer_dim_lst):
            if self.batch_norm:
                deep_output = tf.layers.dense(inputs=deep_output, units=dim, use_bias=False, kernel_initializer=initializer, 
                    kernel_regularizer=regularizer, name='deep_input_' + str(i + 1))
                deep_output = tf.layers.batch_normalization(deep_output, momentum = 0.9, training=self.training)
                deep_output = activation(deep_output)
            else:
                deep_output = tf.layers.dense(inputs=deep_output, units=dim, use_bias=True, activation=activation,
                                          kernel_initializer=initializer, bias_initializer=initializer,
                                          kernel_regularizer=regularizer, name='deep_input_' + str(i + 1))
            # 使用BN后，dropout一般可以省略
            deep_output = tf.layers.dropout(inputs=deep_output, rate=dropout_prob, training=self.training)
        return deep_output

    def _build_model(self):
        # self.initializer = self._get_init_method(self.init_method, self.init_value)
        # self.activation = self._get_activate_method(self.activate_method)
        # self.regularizer = self._get_regularizer(self.reg_method, self.l1_scale, self.l2_scale)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        #self.training = tf.placeholder_with_default(False, shape=(), name='training')
        self.training = tf.placeholder(tf.bool, name='training')
        self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
        deep_input = self._build_input(self.cate_feats, self.cate_feat_val_num, self.dense_feats, self.mult_feats,
                                       self.com_feats_name, self.initializer)
        deep_output = self._build_deep(deep_input, self.layer_dim_lst,
                                       self.activation, self.initializer, self.regularizer, self.keep_prob)
        deep_output = tf.layers.dense(deep_output, 1, activation=tf.nn.sigmoid,
                                           kernel_initializer=self.initializer, bias_initializer=self.initializer,
                                           kernel_regularizer=self.regularizer, name='out_sigmoid')
        self.deep_output = tf.identity(deep_output, name='output')
        self.global_step = tf.Variable(0, trainable=False)
        self.loss = tf.losses.log_loss(self.labels, self.deep_output)
        # regularizer_loss = tf.losses.get_regularization_loss()
        # self.loss += regularizer_loss
        learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=self.global_step,
                                                        decay_steps=self.decay_steps, decay_rate=0.99)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = self._get_opt_method(self.opt_method, learning_rate).minimize(self.loss, self.global_step)

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
        columns = [self.label_name] + self.dense_feats + self.cate_feats #+ self.mult_feats + [self.com_feats_name]
        usecols = range(0, len(columns))
        # cvt test tfrecord 2hour
        train_instance = self._cvt_tfrecord_load_data(train_file, columns, usecols, num_epochs=epochs)
        valid_instance = self._cvt_tfrecord_load_data(valid_file, columns, usecols, num_epochs=1)
        train_batch_size = batch_size*20
        batch_train_instance = tf.train.batch([train_instance],
                                                 batch_size = train_batch_size,
                                                 num_threads = 30,
                                                 capacity = train_batch_size+1024, enqueue_many=False, allow_smaller_final_batch = True)
        '''
        batch_train_instance = tf.train.shuffle_batch([train_instance],
                                                 batch_size = train_batch_size,
                                                 num_threads = 30,
                                                 capacity = train_batch_size+1024, enqueue_many=False, allow_smaller_final_batch = True, min_after_dequeue=batch_size)
        '''

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            loss_sum = 0
            while not coord.should_stop():
                batch_arr = self.sess.run(batch_train_instance)
                #logging.info("---{} {}".format(batch_arr, batch_arr.shape))
                #batch_arr = batch_arr.reshape(train_batch_size, len(columns))
                #batch_df = pd.DataFrame(data=batch_arr, columns=columns)
                #logging.info("---------{} {}".format(batch_df, batch_df.shape))
                #batchs = self._get_batch(batch_arr, batch_size, is_sample=False, is_norm=self.is_norm)
                batchs = self._gen_batch(batch_arr, batch_size)
                for df_batch in batchs:
                    #if df_batch.shape[0] < batch_size: continue
                    loss, step = self._train_batch(df_batch, self.sess)
                    loss_sum += loss
                    if verb > 0 and step % verb == 0:
                        logging.info('{} steps\tloss={}'.format(step, loss_sum / verb))
                        loss_sum = 0 
        except tf.errors.OutOfRangeError:
            logging.info('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

        valid_batch_size = 8192*10
        batch_valid_instance = tf.train.batch([valid_instance],
                                                 batch_size = valid_batch_size,
                                                 num_threads = 50,
                                                 capacity = valid_batch_size+1024, enqueue_many=False, allow_smaller_final_batch = True)
        valid_labels, valid_preds = self._predict_with_queue(batch_valid_instance, valid_batch_size, 0, None, 'valid', 100, columns=columns)
        logging.info("valid instance num {}".format(len(valid_labels)))
        valid_mval = self._evaluate(valid_labels, valid_preds, self.metrics)
        valid_logloss = self._evaluate_logloss(valid_labels, valid_preds, True)
        logging.info('valid-{}: {}\tvalid-logloss: {}'.format(
            self.metrics_name, valid_mval, valid_logloss
        ))
        df_result = pd.DataFrame({'valid_preds': valid_preds})
        logging.info('valid predict infos:\n{}'.format(df_result.describe()))
        logging.info('saving model to file')
        self.save(model_dir)

    def _train_batch(self, arr_train, sess):
        feed_dict = {self.labels:arr_train[:, 0:1], self.keep_prob:self.drop_out, self.training:True}
        prev_col_num = 1 # label
        if self.is_use_dense > 0:
            feed_dict[self.dense_input] = arr_train[:, prev_col_num:len(self.dense_feats)+prev_col_num]
            prev_col_num += len(self.dense_feats)

        for index, col in enumerate(self.cate_feats):
            #print "++", arr_train[:, prev_col_num+index:prev_col_num+index+1], arr_train[:, prev_col_num+index:prev_col_num+index+1].shape
            feed_dict[self.cate_input_lst[index]] = arr_train[:, prev_col_num+index:prev_col_num+index+1]
        # feed_dict[self.mult_input] = df_train[self.mult_feats].values
            # print df_train[[mult]].values
        # feed_dict[self.com_input] = df_train[[self.com_feats_name]].values
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

    def _cvt_tfrecord_load_data(self, data_file, columns, usecols, num_epochs=2):
        success_file = data_file+'.SUCCESS'
        fth_file = data_file + '.fth'
        if not os.path.exists(success_file):
            logging.info('convert csv file {} to tfrecord'.format(data_file))
            if not os.path.exists(fth_file):
                df_tmp = pd.read_csv(data_file, sep='\t', names=columns, usecols=usecols)
                feather.write_dataframe(df_tmp, fth_file)
            df_tmp = feather.read_dataframe(fth_file, columns=columns, use_threads=True)

            def serialize_example(rec_arr, columns):
                #feature = {fea:tf.train.Feature(float_list=tf.train.FloatList(value=[rec_arr[i]])) for i, fea in enumerate(columns)} # 速度太慢
                #feature = {
                #    'fea':tf.train.Feature(bytes_list=tf.train.BytesList(value=[rec_arr[1:].tostring()])),
                #    'label':tf.train.Feature(float_list=tf.train.FloatList(value=[rec_arr[0]])),
                #}
                feature = {
                    'instance':tf.train.Feature(bytes_list=tf.train.BytesList(value=[rec_arr[0:].tostring()])),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                return example_proto.SerializeToString()

            tfrecord_batch_size = 8192*150
            def func_write_tfrecord(tfrec_file, df, columns):
                linecnt = 0
                writer = tf.io.TFRecordWriter(tfrec_file)
                for rec_arr in df.values:
                    example = serialize_example(rec_arr, columns)
                    writer.write(example)
                    linecnt += 1
                    if linecnt % 10000 == 0:
                        logging.info("write_tfrecord {} {}".format(tfrec_file, linecnt))
                writer.close()

            buff = []
            thread_batch_num = 0
            tfrecord_batch_num = df_tmp.shape[0] / tfrecord_batch_size + 1
            for i in xrange(tfrecord_batch_num):
                bg = i * tfrecord_batch_size
                ed = bg + tfrecord_batch_size
                df_batch = df_tmp.iloc[bg:ed]
                tfrec_file = data_file+"."+str(i)+".tfrecord"
                buff.append((tfrec_file, df_batch, columns))
                thread_batch_num += 1
                if thread_batch_num == 50:
                    t_arr = []
                    for i in xrange(thread_batch_num):
                        t_arr.append(threading.Thread(target=func_write_tfrecord, args=(buff[i][0], buff[i][1], buff[i][2], )))
                    for i in xrange(thread_batch_num):
                        t_arr[i].start()
                    for i in xrange(thread_batch_num):
                        t_arr[i].join()
                    buff = []
                    thread_batch_num = 0
            if thread_batch_num > 0 and len(buff) > 0:
                t_arr = []
                for i in xrange(thread_batch_num):
                    t_arr.append(threading.Thread(target=func_write_tfrecord, args=(buff[i][0], buff[i][1], buff[i][2], )))
                for i in xrange(thread_batch_num):
                    t_arr[i].start()
                for i in xrange(thread_batch_num):
                    t_arr[i].join()

            os.system("touch {}".format(success_file))
        filename_queue = data_file+".*.tfrecord"
        logging.info('tfrecord data file {}'.format(filename_queue))
        train_tfrecord_files = tf.train.match_filenames_once(filename_queue)
        #self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        train_tfrecord_files_arr = self.sess.run([train_tfrecord_files])
        logging.info("load file list {}".format(train_tfrecord_files_arr[0]))
        #train_test_type = "train" if "train" in data_file else "test"
        #train_tfrecord_files = ["data/"+f for f in os.listdir("data/") if train_test_type in f and ".tfrecord" in f]
        filename_queue = tf.train.string_input_producer(train_tfrecord_files, num_epochs=num_epochs, capacity=len(train_tfrecord_files_arr[0]))
        _, serialized_example = tf.TFRecordReader().read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'instance': tf.FixedLenFeature([], tf.string),
                                           })
        #label = tf.cast(features["label"], tf.float32)
        #fea = tf.decode_raw(features["fea"], tf.float64) # 注意这里一定要取tf.float64
        #fea = tf.reshape(fea, [1, len(columns)-1])
        #fea = tf.reshape(fea, [len(columns)-1, 1])
        instance = tf.decode_raw(features["instance"], tf.float64) # 注意这里一定要取tf.float64
        instance = tf.reshape(instance, [len(columns), ])
        #instance = tf.reshape(instance, [1, len(columns), ])
        return instance

    def _predict_with_queue(self, batch_instance, batch_size, block_num, metrics=None, pred_type='test', verb=100, columns=None):
        if pred_type not in ['test', 'train', 'valid']:
            raise ValueError("pred_type not in ['test', 'train', 'valid']")
        if pred_type == 'test':
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        preds = list()
        labels = list()
        num = 0
        try:
            while not coord.should_stop():
                batch_arr = self.sess.run(batch_instance)
                #print "---", batch_arr, batch_arr.shape
                tmp_preds = self._predict_batch(batch_arr, self.sess)
                preds += tmp_preds
                labels += batch_arr[:, 0].tolist()
                num += 1
                if verb > 0 and num % verb == 0:
                    logging.info('already predicted {} data'.format(num * batch_size))
        except tf.errors.OutOfRangeError:
            logging.info('Done predicting -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

        preds = np.concatenate(preds).reshape(-1,)
        if metrics:
            mval = self._evaluate(labels, preds, metrics)
            logging.info('{}: {}'.format(self.metrics_name, mval))
        return labels, preds

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

    def _predict_batch(self, arr_data, sess):
        feed_dict = {self.keep_prob:0.0, self.training:False}
        prev_col_num = 1 # label
        if self.is_use_dense > 0:
            feed_dict[self.dense_input] = arr_data[:, prev_col_num:len(self.dense_feats)+prev_col_num]
            prev_col_num += len(self.dense_feats)

        for index, col in enumerate(self.cate_feats):
            feed_dict[self.cate_input_lst[index]] = arr_data[:, prev_col_num+index:prev_col_num+index+1]

        # feed_dict[self.mult_input] = df_data[self.mult_feats].values
        # feed_dict[self.com_input] = df_data[[self.com_feats_name]].values
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

    def _get_batch(self, data_arr, batch_size, is_sample=False, is_norm=False):
        batchs = list()
        if is_sample:
            np.random.shuffle(data_arr)
        if is_norm:
            # columns = [self.label_name] + self.dense_feats + self.cate_feats
            for idx in xrange(len(self.dense_feats)):
                dense_idx = idx + 1
                max_val = data_arr[:,dense_idx].max()
                min_val = data_arr[:,dense_idx].min()
                scale = max_val - min_val
                if scale == 0:
                    scale = 1
                data_arr[:,dense_idx] = (data_arr[:,dense_idx] - min_val) / scale
        i = 0
        while True:
            bg = i * batch_size
            if bg >= data_arr.shape[0]:
                break
            ed = bg + batch_size
            arr_batch = data_arr[bg:ed]
            batchs.append(arr_batch)
            i += 1

        return batchs

    def _gen_batch(self, data_arr, batch_size):
        for i in xrange(0, 10000000):
            bg = i * batch_size
            if bg >= data_arr.shape[0]:
                break
            ed = bg + batch_size
            arr_batch = data_arr[bg:ed]
            yield arr_batch

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




