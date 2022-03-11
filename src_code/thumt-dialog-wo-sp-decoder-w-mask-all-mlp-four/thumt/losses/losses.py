# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import code

def smoothed_softmax_cross_entropy_with_logits(**kwargs):
    logits = kwargs.get("logits")
    labels = kwargs.get("labels")
    smoothing = kwargs.get("smoothing") or 0.0
    normalize = kwargs.get("normalize")
    scope = kwargs.get("scope")

    if logits is None or labels is None:
        raise ValueError("Both logits and labels must be provided")

    with tf.name_scope(scope or "smoothed_softmax_cross_entropy_with_logits",
                       values=[logits, labels]):

        labels = tf.reshape(labels, [-1])

        if not smoothing:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.cast(logits, tf.float32),
                labels=labels
            )
            return ce

        # label smoothing
        vocab_size = tf.shape(logits)[1]

        n = tf.to_float(vocab_size - 1)
        p = 1.0 - smoothing
        q = smoothing / n

        soft_targets = tf.one_hot(tf.cast(labels, tf.int32), depth=vocab_size,
                                  on_value=p, off_value=q)
        soft_targets = tf.stop_gradient(soft_targets)
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=tf.cast(logits, tf.float32),
            labels=soft_targets)

        if normalize is False:
            return xentropy

        # Normalizing constant is the best cross-entropy value with soft
        # targets. We subtract it just for readability, makes no difference on
        # learning
        normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

        return xentropy - normalizing


def smoothed_sigmoid_cross_entropy_with_logits(**kwargs):
    logits = kwargs.get("logits")
    labels = kwargs.get("labels")
    #tes = kwargs.get("tes")
    scope = kwargs.get("scope")

    if logits is None or labels is None:
        raise ValueError("Both logits and labels must be provided")

    with tf.name_scope(scope or "smoothed_sigmoid_cross_entropy_with_logits",
                       values=[logits, labels]):

        #labels = tf.reshape(labels, [-1])

        # label smoothing
        vocab_size = tf.shape(logits)[1]
        #print("vocab_size", tf.shape(vocab_size))
        #print("logits", tf.shape(logits))
        #print("labels", tf.shape(labels))
        #print("tes", tf.shape(tes))

        multi_one_hot = tf.map_fn(lambda x: tf.one_hot(tf.cast(x, tf.int32), depth=vocab_size), labels, dtype = tf.float32)
        soft_targets = tf.reduce_max(multi_one_hot, axis = 1)

        #print("multi_one_hot:", tf.shape(multi_one_hot))
        #print("soft_targets", tf.shape(soft_targets))
        #code.interact(local=locals())



        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                           labels=soft_targets)
        return xentropy

def bag_of_words_loss(bow_logits, target_bow, weight=None):
    ''' Calculate bag of words representation loss
    Args
        - bow_logits: [num_sentences, vocab_size]
        - target_bow: [num_sentences]
    '''
    log_probs = F.log_softmax(bow_logits, dim=1)
    target_distribution = target_bow / (target_bow.sum(1).view(-1, 1) + 1e-23) + 1e-23
    entropy = -(torch.log(target_distribution) * target_bow).sum()
    loss = -(log_probs * target_bow).sum() - entropy

    return loss
