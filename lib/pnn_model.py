# -*- coding: utf-8 -*-
import codecs
import tensorflow as tf
import numpy as np
import std_inputfn_config
import metric_evaluation
from tensorflow.python import debug as tf_debug


class PNNModel(object):
    def __init__(self, train_files, test_files, clf_file, file_read_config_param, model_dir):
        self.train_files = train_files
        self.test_files = test_files
        self.clf_file = clf_file
        self.file_read_config_param = file_read_config_param
        self.model_dir = model_dir
        self.train_input_config = None
        self.test_input_config = None

        self.feature_infos = std_inputfn_config.load_feature_infos(clf_file)
        self._load_train_test_input_config(file_read_config_param)

        model_params = {'dense_feature_names': self.train_input_config.c_columns, 
                        'cate_feature_names':self.train_input_config.d_columns,
                        'mult_feature_names':self.train_input_config.mult_columns,
                        'hidden_units':file_read_config_param['hidden_units'],
                        'n_classes':file_read_config_param['n_classes'],
                        'decay_steps':file_read_config_param['decay_steps'],
                        'learning_rate':file_read_config_param['learning_rate'],}

        # GPU显存按需分配
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.run_config = tf.estimator.RunConfig(save_checkpoints_steps=10000).replace(session_config=gpu_config)

        self.model = tf.estimator.Estimator(model_fn=self.model_fn,
                               params=model_params,
                               model_dir=self.model_dir, config=self.run_config)

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
                                         batch_size=file_read_config_param['batch_size']*20, 
                                         perform_shuffle=False, drop_remainder=False)

    def _get_emb_dim(self, val_num):
        log_val = np.log2(val_num + 1)
        dim = int(log_val)
        if log_val - dim > 0.1:
            dim += 1
        return dim
    def _get_emb_dim(self):
        return 32

    def model_fn(self, features, labels, mode, params):
        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
        elif mode == tf.estimator.ModeKeys.EVAL:
            tf.logging.info("my_model_fn: EVAL, {}".format(mode))
        elif mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

        # 1. build features
        tf.logging.info("----- building features -----")
        dense_feature_num, cate_feature_num = len(params['dense_feature_names']), len(params['cate_feature_names'])
        feature_columns_dense = [tf.feature_column.numeric_column(col_name) for col_name in params['dense_feature_names']]
        feature_columns_cate= [tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity(col_name, num_buckets=cate_fea_num+1, default_value=0), \
            dimension=self._get_emb_dim()) for col_name, cate_fea_num in zip(params['cate_feature_names'], self.feature_infos['cate_feat_val_num'])]
        feature_columns_mult = []
        if len(params["mult_feature_names"]) > 0:
            feature_columns_mult = [tf.feature_column.numeric_column(col_name, shape=(self.train_input_config.mult_feature_val_max_num,), default_value=0, dtype=tf.int32) \
                for col_name in params['mult_feature_names']]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # 2. build forward propagation and [PREDICT] mode
        with tf.name_scope("input"):
            input_layer_dense = tf.feature_column.input_layer(features, feature_columns_dense)
            input_layer_cate = tf.feature_column.input_layer(features, feature_columns_cate)
            print "---------- raw_input_layer_dense", input_layer_dense
            print "---------- raw_input_layer_cate", input_layer_cate

            if len(params["mult_feature_names"]) > 0:
                input_layer_mult = tf.feature_column.input_layer(features, feature_columns_mult)
                input_layer_mult = tf.cast(input_layer_mult, tf.int32)

        if len(params["mult_feature_names"]) > 0:
            with tf.name_scope("mult_embedding"):
                emb_table_mult = tf.get_variable(name='{}_emb'.format("mult"),
                                           shape=[self.feature_infos['mult_feat_val_num'][0]+1, self._get_emb_dim()],
                                           initializer=tf.glorot_normal_initializer(),
                                           dtype=tf.float32)
                emb_mult = tf.nn.embedding_lookup(emb_table_mult, input_layer_mult) #512*20*k
                sparse_idx = tf.where(tf.greater(input_layer_mult, 0))
                mask_input_layer_mult = tf.cast(tf.greater(input_layer_mult, 0),dtype=tf.float32) #512*20
                mask_emb_mult = tf.multiply(emb_mult, tf.expand_dims(mask_input_layer_mult, -1)) #512*20*k

                nonzero_num = tf.reduce_sum(mask_input_layer_mult, axis=1) #512*1
                nonzero_num = tf.where(nonzero_num>0, nonzero_num, tf.ones_like(nonzero_num))
                nonzero_num = tf.reshape(nonzero_num, [-1, 1])

                average_emb_mult = tf.reshape(tf.reduce_sum(mask_emb_mult, axis=1), [-1, self._get_emb_dim()]) / nonzero_num

        cate_feature_emb = tf.reshape(input_layer_cate, [-1, cate_feature_num, self._get_emb_dim()])
        print "----------", cate_feature_emb
        input_layer_dense = tf.layers.batch_normalization(input_layer_dense, momentum = 0.9, training=is_training)
        input_layer = tf.concat([input_layer_dense, input_layer_cate], axis=1)
        for idx1, batch_emb1 in enumerate(xrange(cate_feature_emb.shape[1])):
            for idx2, batch_emb2 in enumerate(xrange(cate_feature_emb.shape[1])):
                if idx2 <= idx1:continue
                inner_prod = cate_feature_emb[:,idx1,:]*cate_feature_emb[:,idx2,:]
                print "---------- inner_prod", inner_prod
                input_layer = tf.concat([input_layer, inner_prod], axis=1)
                #input_layer = tf.reduce_sum(inner_prod, axis=1)
                print "---------- input_layer2", input_layer
        hidden_units = params['hidden_units']
        if len(params["mult_feature_names"]) > 0:
            deep_output = tf.concat([input_layer, average_emb_mult], axis=1)
        else:
            deep_output = input_layer
        for i, dim in enumerate(hidden_units):
            deep_output = tf.layers.dense(inputs=deep_output, units=dim, use_bias=True, kernel_initializer=tf.compat.v1.glorot_uniform_initializer())
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
        input_fn = self.train_input_config.input_mult_fn if len(self.train_input_config.mult_columns) > 0 else self.train_input_config.input_fn
        self.model.train(input_fn=self.train_input_config.input_fn, steps=None)

    def evaluate(self):
        input_fn = self.test_input_config.input_mult_fn if len(self.train_input_config.mult_columns) > 0 else self.test_input_config.input_fn
        result = self.model.evaluate(input_fn=self.test_input_config.input_fn, steps=None)

    def train_and_evaluate(self):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        input_fn = self.train_input_config.input_mult_fn if len(self.train_input_config.mult_columns) > 0 else self.train_input_config.input_fn
        train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
        input_fn = self.test_input_config.input_mult_fn if len(self.train_input_config.mult_columns) > 0 else self.test_input_config.input_fn
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, throttle_secs=0)
        tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec)

    def predict(self):
        input_fn = self.test_input_config.input_mult_fn if len(self.train_input_config.mult_columns) > 0 else self.test_input_config.input_fn
        result = self.model.predict(input_fn=input_fn)
        return result

