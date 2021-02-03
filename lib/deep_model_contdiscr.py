# -*- coding: utf-8 -*-
import codecs
import tensorflow as tf
import numpy as np
import std_inputfn_config
import metric_evaluation
from tensorflow.python import debug as tf_debug


class DeepModel(object):
    def __init__(self, train_files, test_files, clf_file, file_read_config_param, model_dir):
        self.train_files = train_files
        self.test_files = test_files
        self.clf_file = clf_file
        self.file_read_config_param = file_read_config_param
        self.model_dir = model_dir
        self.train_input_config = None
        self.test_input_config = None

        self.feature_infos = self._load_feature_infos(clf_file)
        self._load_train_test_input_config(file_read_config_param)

        model_params = {'dense_feature_names': self.train_input_config.c_columns, 
                        'cate_feature_names':self.train_input_config.d_columns,
                        'hidden_units':file_read_config_param['hidden_units'],
                        'n_classes':file_read_config_param['n_classes'],
                        'decay_steps':file_read_config_param['decay_steps'],
                        'learning_rate':file_read_config_param['learning_rate'],}

        # GPU显存按需分配
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.run_config = tf.estimator.RunConfig().replace(session_config=gpu_config)

        self.model = tf.estimator.Estimator(model_fn=self.model_fn,
                               params=model_params,
                               model_dir=self.model_dir, config=self.run_config)

    def _load_feature_infos(self, feature_classify_file):
        #import logging
        cate_feats = list()
        cate_feat_val_num = list()
        dense_feats = list()
        mult_feats = list()
        mult_val_num = 1500000
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
                    mult_val_num = int(fs[3])
        print('cate_feats size: {}, cate_feat_val_num size: {}, dense_feats size: {}, mult_feats size: {}'.format(
            len(cate_feats), len(cate_feat_val_num), len(dense_feats), len(mult_feats)
        ))
        return {'cate_feats':cate_feats, 'cate_feat_val_num':cate_feat_val_num, 'dense_feats':dense_feats, 'mult_feats':mult_feats, 'mult_val_num':mult_val_num}

    def _load_train_test_input_config(self, file_read_config_param):
        field_delim = file_read_config_param['field_delim']
        dense_feats_num = len(self.feature_infos['dense_feats'])
        cate_feats_num = len(self.feature_infos['cate_feats'])
        self.train_input_config = std_inputfn_config.StandardInputFnConfig(self.train_files, field_delim, self.clf_file, 
                                         continuous_feature_number=dense_feats_num,
                                         discrete_feature_number=cate_feats_num,
                                         num_epochs=file_read_config_param['epoch'],
                                         batch_size=file_read_config_param['batch_size'], 
                                         buffer_size=file_read_config_param['buffer_size'],
                                         perform_shuffle=True, drop_remainder=True)

        self.test_input_config = std_inputfn_config.StandardInputFnConfig(self.test_files, field_delim, self.clf_file, 
                                         continuous_feature_number=dense_feats_num,
                                         discrete_feature_number=cate_feats_num,
                                         num_epochs=1,
                                         batch_size=file_read_config_param['batch_size']*10, 
                                         perform_shuffle=False, drop_remainder=False)

    def _get_emb_dim(self, val_num):
        log_val = np.log2(val_num + 1)
        dim = int(log_val)
        if log_val - dim > 0.1:
            dim += 1
        return dim
    def _get_emb_dim(self):
        return 64

    def model_fn(self, features, labels, mode, params):
        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
        elif mode == tf.estimator.ModeKeys.EVAL:
            tf.logging.info("my_model_fn: EVAL, {}".format(mode))
        elif mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

        # 1. build features
        tf.logging.info("----- building features -----")
        feature_columns = [tf.feature_column.numeric_column(col_name) for col_name in params['dense_feature_names']]
        #feature_columns += [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(col_name, num_buckets=cate_fea_num+1, default_value=0)) \
        #    for col_name, cate_fea_num in zip(params['cate_feature_names'], self.feature_infos['cate_feat_val_num'])]
        feature_columns += [tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity(col_name, num_buckets=cate_fea_num+1, default_value=0), \
            dimension=self._get_emb_dim()) for col_name, cate_fea_num in zip(params['cate_feature_names'], self.feature_infos['cate_feat_val_num'])]
            #dimension=self._get_emb_dim(cate_fea_num)) for col_name, cate_fea_num in zip(params['cate_feature_names'], self.feature_infos['cate_feat_val_num'])]

        feature_columns_bucketized = []
        # bucketized  col1:1,3,5,7;col2:2,4,6
        bucketized = params.get("bucketized")
        if bucketized != None:
            for colname_bound in bucketized.split(";"):
                colname, bound = colname_bound.split(":")
                bound = sorted([int(n) for n in bound.split(",")])
                feature_columns_bucketized.append(tf.feature_column.bucketized_column(tf.feature_column.numeric_column(colname, default_value=0.0), boundaries=bound))
        feature_columns += feature_columns_bucketized

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # 2. build forward propagation and [PREDICT] mode
        with tf.name_scope("input"):
            input_layer = tf.feature_column.input_layer(features, feature_columns)
            input_layer = tf.layers.batch_normalization(input_layer, momentum = 0.9, training=is_training)
        #print sess.run(input_layer).shape
        hidden_units = params['hidden_units']
        deep_output = input_layer
        for i, dim in enumerate(hidden_units):
            deep_output = tf.layers.dense(inputs=deep_output, units=dim, use_bias=True, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01))
            deep_output = tf.nn.relu(deep_output)
            deep_output = tf.layers.batch_normalization(deep_output, momentum = 0.9, training=is_training)
        logits = tf.layers.dense(deep_output, params['n_classes'], activation=None, use_bias=True) # N*n_classes
        predictions = { 'class_ids': tf.argmax(input=logits, axis=1),
                        'probabilities':tf.nn.softmax(logits), }  # self-define predict output

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # 3. build loss and [EVAL] mode
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) # 使用这个会报错
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])
        #print sess.run(predictions)
        #tf.logging.info("-----{},{},{}".format(labels, predictions['probabilities'], predictions['probabilities'][:,1]))

        ctr_auc, auc, ctcvr_auc, ctr_logloss, ctcvr_logloss = -1, -1, -1, -1, -1
        if params['n_classes'] == 2:
            ctr_auc, ctr_logloss = metric_evaluation.calc_auc_logloss(labels, predictions['probabilities'])
        elif params['n_classes'] == 3:
            ctr_auc, ctcvr_auc, ctr_logloss, ctcvr_logloss = metric_evaluation.calc_cvr_auc_logloss(labels, predictions['probabilities'])
        #print "------------------------", ctr_auc, ctcvr_auc, ctr_logloss, ctcvr_logloss
        #logging_hook = tf.estimator.LoggingTensorHook(tensors={"loss" : loss, "ctr_auc":ctr_auc, "prob":predictions['probabilities'][0,1]}, every_n_iter=10, at_end=False)
        if mode == tf.estimator.ModeKeys.EVAL:
            if params['n_classes'] == 2:
                return tf.estimator.EstimatorSpec(
                        mode,
                        loss=loss,
                        eval_metric_ops={'my_accuracy': accuracy, 'ctr_auc':ctr_auc, 'ctr_logloss':ctr_logloss},
                )
            elif params['n_classes'] == 3:
                return tf.estimator.EstimatorSpec(
                        mode,
                        loss=loss,
                        eval_metric_ops={'my_accuracy': accuracy, 'ctr_auc':ctr_auc, 'ctcvr_auc':ctcvr_auc, 'ctr_logloss':ctr_logloss, 'ctcvr_logloss':ctcvr_logloss},
                )

        # 4. build optimizer and [TRAIN] mode
        assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"
        learning_rate = tf.train.exponential_decay(learning_rate=params['learning_rate'], global_step=tf.train.get_global_step(),
                                                        decay_steps=params.get('decay_steps', 500), decay_rate=0.99)
        #optimizer = tf.train.AdagradOptimizer(params['learning_rate'])
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        tf.summary.scalar('my_accuracy', accuracy[1])
        tf.summary.scalar('learning_rate', learning_rate)
        return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op, 
                #training_hooks = [logging_hook]
        )

    def train(self):
        #self.model.train(input_fn=self.train_input_config.input_fn, hooks=hooks, steps=2000)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        #tensors_to_log = {'ctr_auc':'ctr_auc/update_op', 'ctcvr_auc':'ctcvr_auc/update_op', 'ctr_logloss':'ctr_logloss/update_op', 'ctcvr_logloss':'ctcvr_logloss/update_op'}
        #logging_hook = tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=2000, at_end=False) #与hook使用，at_end必须为False
        #self.model.train(input_fn=self.train_input_config.input_fn, steps=None, hooks=[logging_hook])
        self.model.train(input_fn=self.train_input_config.input_fn, steps=None)

    def evaluate(self):
        #hooks = [tf_debug.LocalCLIDebugHook()]
        result = self.model.evaluate(input_fn=self.test_input_config.input_fn, steps=None)

    def predict(self):
        result = self.model.predict(input_fn=self.test_input_config.input_fn)
        return result

