
import tensorflow as tf

def my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def calc_auc_logloss(labels, predictions):
        ck_label = tf.where(labels>=1, tf.ones_like(labels), tf.zeros_like(labels))
        ctr_auc = tf.compat.v1.metrics.auc(ck_label, predictions[:,1], name='ctr_auc')
        ctr_logloss = tf.compat.v1.metrics.mean(tf.losses.log_loss(labels=ck_label, predictions=predictions[:,1]), name='ctr_logloss')
        return ctr_auc, ctr_logloss

def calc_cvr_auc_logloss(labels, predictions):
    # labels:0 predictions:[0.2,0.5,0.3]
    ck_label = tf.where(labels>=1, tf.ones_like(labels), tf.zeros_like(labels))
    ck_predictions = 1 - predictions[:,0]
    #ck_predictions = tf.add(predictions[:,1], predictions[:,2])
    #ck_predictions = my_tf_round(ck_predictions, 6)
    #ck_predictions = tf.where(ck_predictions>1, tf.ones_like(ck_predictions), ck_predictions)

    ctr_auc = tf.compat.v1.metrics.auc(ck_label, ck_predictions, name='ctr_auc')
    ctr_logloss = tf.compat.v1.metrics.mean(tf.losses.log_loss(labels=ck_label, predictions=ck_predictions), name='ctr_logloss')

    ctcv_label = tf.where(labels>=2, tf.ones_like(labels), tf.zeros_like(labels))
    ctcv_predictions = predictions[:,2]
    ctcv_predictions = tf.where(ctcv_predictions>1, tf.ones_like(ctcv_predictions), ctcv_predictions)

    ctcvr_auc = tf.compat.v1.metrics.auc(ctcv_label, ctcv_predictions, name='ctcvr_auc')
    ctcvr_logloss = tf.compat.v1.metrics.mean(tf.losses.log_loss(labels=ctcv_label, predictions=ctcv_predictions), name='ctcvr_logloss')
    return ctr_auc, ctcvr_auc, ctr_logloss, ctcvr_logloss

