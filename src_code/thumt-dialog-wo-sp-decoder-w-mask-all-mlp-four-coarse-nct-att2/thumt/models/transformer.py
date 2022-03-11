# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy,code
import numpy as np
import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils as utils

from thumt.models.model import NMTModel


def _layer_process(x, mode, trainable=True):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x, trainable=trainable)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None, trainable=True):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True, trainable=trainable)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True, trainable=trainable)

        return output

def _load_embedding(mode, params, uniform_scale = 0.25, dimension_size = 300, embed_file='glove'):
    n = 0
    word_vectors = np.memmap("/path/to/coarse_features/all.features.mmap", mode="r", dtype=np.float32, shape=(142871, 1000))
#    word_vectors = np.memmap("/path/to/coarse_features/all.features.mmap", mode="r", dtype=np.float32, shape=(30370, 1000))
    return np.array(word_vectors, dtype=np.float32)

def birnn(inputs, sequence_length, params):
    lstm_fw_cell = rnn.BasicLSTMCell(params.hidden_size)
    lstm_bw_cell = rnn.BasicLSTMCell(params.hidden_size)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs,
                                                 sequence_length=sequence_length, dtype=tf.float32)
    states_fw, states_bw = outputs
    return tf.concat([states_fw, states_bw], axis=2)

def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z= mu + tf.multiply(std, epsilon)
    return z

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld

def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    input_tensor: float Tensor to perform activation.
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf

def transformer_context(inputs, bias, params, dtype=None, scope="ctx_transformer", trainable=True):
    with tf.variable_scope(scope, default_name="context", dtype=dtype,
                           values=[inputs, bias], reuse=tf.AUTO_REUSE):
        x = inputs
        for layer in range(params.num_context_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        trainable=trainable
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        trainable=trainable
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs

def transformer_encoder(inputs, bias, params, dia_mask=None, dtype=None, scope=None, trainable=True, get_first_layer=False):
    with tf.variable_scope("encoder", dtype=dtype,
                           values=[inputs, bias], reuse=tf.AUTO_REUSE):
        x = inputs
        for layer in range(params.num_encoder_layers):
            if layer < params.bottom_block:
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        max_relative_dis = params.max_relative_dis \
                            if params.position_info_type == 'relative' else None
    
                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            None,
                            bias,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            max_relative_dis=max_relative_dis,
                            trainable=trainable
                        )
                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                    with tf.variable_scope("feed_forward"):
                        y = _ffn_layer(
                            _layer_process(x, params.layer_preprocess),
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                            trainable=trainable
                        )
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)
                first_layer_output = x
                #print("first_layer_output", first_layer_output)
                if get_first_layer and layer == (params.bottom_block - 1):
                    return x, first_layer_output
            else:
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        max_relative_dis = params.max_relative_dis \
                            if params.position_info_type == 'relative' else None

                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            None,
                            bias,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            max_relative_dis=max_relative_dis,
                            trainable=trainable,
                            dia_mask=dia_mask
                        )
                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                    with tf.variable_scope("feed_forward"):
                        y = _ffn_layer(
                            _layer_process(x, params.layer_preprocess),
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                            trainable=trainable
                        )
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)

#            if params.bottom_block and get_first_layer:
#                return first_layer_output, first_layer_output

        outputs = _layer_process(x, params.layer_preprocess)
        if params.bottom_block == 0:
            first_layer_output = x

        return outputs, first_layer_output


def transformer_decoder(inputs, memory, image_output, bias, mem_bias, params, state=None, dia_mask=None, 
                        dtype=None, scope=None, trainable=True):
    with tf.variable_scope("decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias], reuse=tf.AUTO_REUSE):
#    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
#                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None
                max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state,
                        max_relative_dis=max_relative_dis,
                        trainable=trainable
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        max_relative_dis=max_relative_dis,
                        trainable=trainable,
                        dia_mask=dia_mask
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        trainable=trainable
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=trainable)
                if image_output is not None:
                    with tf.variable_scope("encdec_image_attention"):
                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            image_output,
                            mem_bias,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            max_relative_dis=max_relative_dis,
                         )
                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess)

                    with tf.variable_scope("image_feed_forward"):
                        y = _ffn_layer(
                            _layer_process(x, params.layer_preprocess),
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                        )
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs


def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size
    src_seq = features["source"]
    sample_seq = features["sample"]
    print(features)
    ctx_dia_src_seq = features["context_dia_src"]
    ctx_dia_tgt_seq = features["context_dia_tgt"]
    ctx_src_seq = features["context_source"]
    sample_seq = features["sample"]

    #emotion = features["emotion"]
    src_len = features["source_length"]
    sample_len = features["sample_length"]

    ctx_dia_src_len = features["context_dia_src_length"]
    ctx_dia_tgt_len = features["context_dia_tgt_length"]
    ctx_src_len = features["context_source_length"]
    sample_len = features["sample_length"]


    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    ctx_dia_src_mask = tf.sequence_mask(ctx_dia_src_len,
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=tf.float32)
    ctx_dia_tgt_mask = tf.sequence_mask(ctx_dia_tgt_len,
                                maxlen=tf.shape(features["context_dia_tgt"])[1],
                                dtype=tf.float32)

    ctx_src_mask = tf.sequence_mask(ctx_src_len,
                                maxlen=tf.shape(features["context_source"])[1],
                                dtype=tf.float32)
    sample_mask = tf.sequence_mask(sample_len,
                                maxlen=tf.shape(features["sample"])[1],
                                dtype=tf.float32)
#    code.interact(local=locals())
    # dialogue mask
    ctx_mask = tf.sequence_mask(ctx_dia_src_len,
                                maxlen=tf.shape(features["context_source"])[1],
                                dtype=tf.float32)
    top_mask = tf.equal(ctx_mask, ctx_src_mask)
    top_mask = tf.cast(top_mask, tf.float32)
    inf = -1e9
    if dtype is None:
        dtype = tf.float32
    if dtype != tf.float32:
        inf = dtype.min
    #dia_mask = inf * top_mask

#    dia_mask = features["mask"]
    length = tf.shape(top_mask)[-1]
    k_mask = tf.tile(top_mask, [1, length])
    ex_mask = tf.expand_dims(k_mask, 1)
    ti_mask = tf.tile(ex_mask, [1, params.num_heads, 1])
    ori_mask = tf.reshape(ti_mask, [-1, params.num_heads, length, length])
    trans_mask = tf.transpose(ori_mask, [0, 1, 3, 2])
    part_mask = ori_mask + trans_mask
    part_mask = tf.cast(part_mask, dtype)
    dia_mask = inf * part_mask

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            src_embedding = tf.get_variable("weights",
                                            [src_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer, trainable=True)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [src_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer, trainable=True)

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)
    #emotion_inputs = tf.gather(src_embedding, emotion)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)
    #    emotion_inputs = emotion_inputs * (hidden_size ** 0.5)

    with tf.variable_scope("img_embedding", reuse=tf.AUTO_REUSE):
        weights_initializer = tf.constant_initializer(_load_embedding(mode, params))
#        img_emb = tf.get_variable(name='embedding_weights', shape=(142871, 1000), initializer=weights_initializer, trainable=False)
        img_emb = tf.get_variable(name='embedding_weights', shape=(30370, 1000), initializer=weights_initializer, trainable=False)

#    img_input = tf.nn.embedding_lookup(img_emb, features["src_image"])

    with tf.variable_scope("turn_position_embedding", reuse=tf.AUTO_REUSE):
        pos_emb = tf.get_variable("turn_pos_embedding", [len(params.vocabulary["position"]), hidden_size], initializer=tf.contrib.layers.xavier_initializer())

    inputs = inputs * tf.expand_dims(src_mask, -1) #src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    #print("img_input", img_input, encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    if params.position_info_type == 'absolute':
        encoder_input = layers.attention.add_timing_signal(encoder_input)
    #segment embeddings
    if params.segment_embeddings:
        seg_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_dia_src"])
        encoder_input += seg_pos_emb
    '''
    multimodal = tf.concat([encoder_input, img_input], -1)
    with tf.variable_scope("for_src_image", reuse=tf.AUTO_REUSE):
        encoder_input = tf.layers.dense(multimodal, hidden_size, activation=tf.nn.tanh)
    '''
    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output, first_layer_output = transformer_encoder(encoder_input, enc_attn_bias, params)

    turn_ctx_src_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_ctx_src"])
    ctx_inputs = tf.gather(src_embedding, ctx_src_seq) * (hidden_size ** 0.5)
    ctx_inputs = ctx_inputs * tf.expand_dims(ctx_src_mask, -1)

    context_input = tf.nn.bias_add(ctx_inputs, bias)
    context_input = layers.attention.add_timing_signal(context_input)
    if turn_ctx_src_pos_emb.shape.as_list()[-2] == context_input.shape.as_list()[-2]:
        context_input = turn_ctx_src_pos_emb + context_input
    #context_input = context_input + turn_ctx_src_pos_emb
    ctx_src_img_input = tf.nn.embedding_lookup(img_emb, features["ctx_src_image"])
    ctx_src_img_input = tf.layers.dense(ctx_src_img_input, hidden_size, activation=tf.nn.tanh)
    ctx_src_img_input = tf.nn.dropout(ctx_src_img_input, 1.0 - params.embed_dropout)
#    image_input = tf.tile(tf.expand_dims(img_input, -2), [1, length, 1])
#    multimodal = tf.concat([context_input, ctx_src_img_input], -1)
#    with tf.variable_scope("for_ctx_src_image", reuse=tf.AUTO_REUSE):
#        context_input = tf.layers.dense(multimodal, hidden_size, activation=tf.nn.tanh)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        context_input = tf.nn.dropout(context_input, keep_prob)

    ctx_attn_bias = layers.attention.attention_bias(ctx_src_mask, "masking")
#        context_lan_src = transformer_context(context_input, ctx_attn_bias, params)
    context_source, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask)

    ## context
    # ctx_seq: [batch, max_ctx_length]
    print("building context graph")
    if params.context_representation == "self_attention":
        print('use self attention')
        # dialogue src context
        get_first_layer = True
    #    dia_mask = None
        turn_dia_src_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_dia_src"])
        
        ctx_inputs = tf.gather(src_embedding, ctx_dia_src_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_dia_src_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_dia_src_pos_emb
        '''
        src_ctx_img_input = tf.nn.embedding_lookup(img_emb, features["src_ctx_image"])
        multimodal = tf.concat([context_input, src_ctx_img_input], -1)
        with tf.variable_scope("for_src_ctx_image", reuse=tf.AUTO_REUSE):
             context_input = tf.layers.dense(multimodal, hidden_size, activation=tf.nn.tanh)
        '''
        context_input = tf.nn.dropout(context_input, 1.0 - params.residual_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_dia_src_mask, "masking")
#        context_dia_src = transformer_context(context_input, ctx_attn_bias, params)
#        print("dia_mask")
        context_dia_src, _ = transformer_encoder(context_input, ctx_attn_bias, params, get_first_layer)
        
#        context_dia_src = first_layer_output

        # dialogue tgt context
        turn_dia_tgt_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_dia_tgt"])
        ctx_inputs = tf.gather(tgt_embedding, ctx_dia_tgt_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_dia_tgt_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_dia_tgt_pos_emb
        '''
        tgt_ctx_img_input = tf.nn.embedding_lookup(img_emb, features["tgt_ctx_image"])
        multimodal = tf.concat([context_input, tgt_ctx_img_input], -1)
        with tf.variable_scope("for_tgt_ctx_image", reuse=tf.AUTO_REUSE):
            context_input = tf.layers.dense(multimodal, hidden_size, activation=tf.nn.tanh)
        '''
        context_input = tf.nn.dropout(context_input, 1.0 - params.residual_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_dia_tgt_mask, "masking")
#        context_dia_tgt = transformer_context(context_input, ctx_attn_bias, params)
        context_dia_tgt, _ = transformer_encoder(context_input, ctx_attn_bias, params, get_first_layer)

        # sample
        sa_inputs = tf.gather(tgt_embedding, sample_seq) * (hidden_size ** 0.5)
        sa_inputs = sa_inputs  * tf.expand_dims(sample_mask, -1)
        sa_input = tf.nn.bias_add(sa_inputs, bias)
        sa_input  = layers.attention.add_timing_signal(sa_input)
        sa_input = tf.nn.dropout(sa_input, 1.0 - params.residual_dropout)
        sa_attn_bias = layers.attention.attention_bias(sample_mask, "masking")

        sa_tgt, _ = transformer_encoder(sa_input, sa_attn_bias, params, get_first_layer)

        ########## clm
        '''
        turn_clm_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_clm"])
        ctx_inputs = tf.gather(src_embedding, clm_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(clm_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_clm_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(clm_mask, "masking")
#        context_dia_tgt = transformer_context(context_input, ctx_attn_bias, params)
        clm_output, _ = transformer_encoder(context_input, ctx_attn_bias, params)
        '''
    elif params.context_representation == "embedding":
        print('use embedding')
        ctx_inputs = tf.gather(src_embedding, ctx_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_mask, -1)
        context_input = tf.nn.bias_add(ctx_inputs, bias)
        ctx_attn_bias = layers.attention.attention_bias(ctx_mask, "masking")
        context_output = context_input
    elif params.context_representation == "bilstm":
        print('use bilstm')
        ctx_inputs = tf.gather(src_embedding, ctx_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_mask, -1)
        context_input = tf.nn.bias_add(ctx_inputs, bias)
        ctx_attn_bias = layers.attention.attention_bias(ctx_mask, "masking")
        context_output = birnn(context_input, ctx_len, params)
    if mode == "infer":
        dia_mask = top_mask
        return encoder_output, context_dia_src, context_dia_tgt, context_source, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_ctx_src_pos_emb, first_layer_output, sa_tgt, dia_mask, ctx_src_img_input

    return encoder_output, context_dia_src, context_dia_tgt, context_source, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_ctx_src_pos_emb, first_layer_output, sa_tgt, top_mask, ctx_src_img_input


def decoding_graph(features, state, mode, params):
    is_training = True
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0
        is_training = False

    dtype = tf.get_variable_scope().dtype
    tgt_seq = features["target"]
    sample_len = features["sample_length"]    #
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    ctx_dia_src_len = features["context_dia_src_length"]
    ctx_dia_tgt_len = features["context_dia_tgt_length"]
    ctx_src_len = features["context_source_length"]

    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    ctx_dia_src_mask = tf.sequence_mask(ctx_dia_src_len,
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=tf.float32)
    ctx_dia_tgt_mask = tf.sequence_mask(ctx_dia_tgt_len,
                                maxlen=tf.shape(features["context_dia_tgt"])[1],
                                dtype=tf.float32)

    ctx_src_mask = tf.sequence_mask(ctx_src_len,
                                maxlen=tf.shape(features["context_source"])[1],
                                dtype=tf.float32)

    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=dtype or tf.float32)

    sample_mask = tf.sequence_mask(sample_len,
                                maxlen=tf.shape(features["sample"])[1],
                                dtype=dtype or tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)

    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [src_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)
    else:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("target_embedding",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                      initializer=initializer, trainable=True)

    if params.use_mrg:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            mrg_weights = tf.get_variable("rg_softmax", [tgt_vocab_size, hidden_size],
                                      initializer=initializer, trainable=True)

    if params.use_crg:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            crg_weights = tf.get_variable("rg_softmax", [tgt_vocab_size, hidden_size],
                                      initializer=initializer, trainable=True)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)
    
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    ctx_src_attn_bias = layers.attention.attention_bias(ctx_src_mask, "masking", dtype=dtype)
    ctx_dia_tgt_attn_bias = layers.attention.attention_bias(ctx_dia_tgt_mask, "masking", dtype=dtype)
    ctx_dia_src_attn_bias = layers.attention.attention_bias(ctx_dia_src_mask, "masking", dtype=dtype)
    ctx_src_attn_bias = layers.attention.attention_bias(ctx_src_mask, "masking", dtype=dtype)
    sample_attn_bias = layers.attention.attention_bias(sample_mask, "masking", dtype=dtype)

    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal", dtype=dtype)
    if params.use_coherence or params.use_cluts or params.use_clus:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            bias = tf.get_variable("tgt_bias", [hidden_size])
        targets_ = tf.nn.bias_add(targets, bias)
        if params.position_info_type == 'absolute':
            targets_ = layers.attention.add_timing_signal(targets_)

        if params.residual_dropout:
            keep_prob = 1.0 - params.residual_dropout
            targets_ = tf.nn.dropout(targets_, keep_prob)

        tgt_attn_bias = layers.attention.attention_bias(tgt_mask, "masking", dtype=dtype)
        tgt_encoder_output, _ = transformer_encoder(targets_, tgt_attn_bias, params)

    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    if params.position_info_type == 'absolute':
        decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]
#    tgt_encoder_output = transformer_context(decoder_input, dec_attn_bias, params)
#    tgt_encoder_output, first_layer_output_dec = transformer_encoder(decoder_input, dec_attn_bias, params)
#    emo_inputs = state["emotion"]
    turn_dia_src_pos_emb = state["position_dia_src"]
    turn_dia_tgt_pos_emb = state["position_dia_tgt"]
    turn_ctx_src_pos_emb = state["position_ctx_src"]
#    turn_lan_tgt_pos_emb = state["position_lan_tgt"]

    context_dia_src_output = state["context_dia_src"]
    context_dia_tgt_output = state["context_dia_tgt"]

    context_source_output = state["context_source"]
    ctx_src_img_input = state["ctx_src_img_input"]
    first_layer_output = state["first_layer_output"]
    sample_output = state["sample"]
    inf = -1e9
    if dtype is None:
        dtype = tf.float32
    if dtype != tf.float32:
        inf = dtype.min

    dia_mask = state["dia_mask"]
    enc_length = tf.shape(dia_mask)[-1]
#    length = tf.shape(decoder_input)[-2]
#    top_mask = tf.tile(dia_mask, [1, length])
#    ex_mask = tf.expand_dims(top_mask, 1)
#    ti_mask = tf.tile(ex_mask, [1, params.num_heads, 1])
#    ori_mask = tf.reshape(ti_mask, [-1, params.num_heads, length, enc_length])
#    part_mask = tf.cast(ori_mask, dtype)
#    dia_mask = inf * part_mask

    #with tf.variable_scope("emotion_embedding", reuse=True):
    # cvae
    #w_query = tf.get_variable("w_Q", [1, params.num_units], dtype=tf.float32)
    #code.interact(local=locals())
    #encoder_rep = tf.reduce_sum(encoder_output * src_mask, -2) / tf.reduce_sum(src_mask)
#    context_rep = tf.reduce_mean(context_output, -2)
    #prior_encode = tf.concat([encoder_rep, context_rep], axis=-1)
#    code.interact(local=locals())
    #emotion_ = emo_inputs[:,0,:]
    context_dia_src = context_dia_src_output[:, 0, :]
#    context_dia_src = first_layer_output[:,0,:]
    context_dia_tgt = context_dia_tgt_output[:, 0, :]
    
    #prior_encode = tf.concat([encoder_rep, emotion_], axis=-1)
#    emotion_ = tf.tile(tf.expand_dims(emotion_, 1), [1, tf.shape(encoder_output)[-2], 1])
    #encoder_output1 = tf.concat([encoder_output, emotion_], axis=-1)
    #s_mask = tf.transpose(src_mask, [0, 2, 1])
#    s_mask = tf.tile(tf.expand_dims(src_mask, -1), [1, 1, tf.shape(encoder_output)[-1]])
#    s_mask = tf.cast(s_mask, encoder_output.dtype)
#    t_mask = tf.tile(tf.expand_dims(tgt_mask, -1), [1, 1, tf.shape(tgt_encoder_output)[-1]])
#    t_mask = tf.cast(t_mask, encoder_output.dtype)
    s_mask = tf.expand_dims(src_mask, -1)
    t_mask = tf.expand_dims(tgt_mask, -1)

    sample_mask = tf.expand_dims(sample_mask, -1)
    sample_rep = tf.reduce_sum(sample_output * sample_mask, -2) / tf.reduce_sum(sample_mask, -2)
#    code.interact(local=locals())

    if mode != "infer":
        # for MT
        length = tf.shape(decoder_input)[-2]
        top_mask = tf.tile(dia_mask, [1, length])
        ex_mask = tf.expand_dims(top_mask, 1)
        ti_mask = tf.tile(ex_mask, [1, params.num_heads, 1])
        ori_mask = tf.reshape(ti_mask, [-1, params.num_heads, length, enc_length])
        part_mask = tf.cast(ori_mask, dtype)
        dia_mask = inf * part_mask

        decoder_output = transformer_decoder(decoder_input, context_source_output, ctx_src_img_input, 
                                             dec_attn_bias, ctx_src_attn_bias,
                                             params, dia_mask=dia_mask)

#        decoder_output = transformer_decoder(decoder_input, encoder_output,
#                                             dec_attn_bias, enc_attn_bias,
#                                             params)
        #tgt_rep = tf.reduce_sum(tgt_encoder_output * t_mask, -2) / tf.reduce_sum(t_mask, -2)
        if params.use_mrg: # monolingual response generation task
            mrg_decoder_output = transformer_decoder(decoder_input, context_dia_tgt_output, None,
                                                 dec_attn_bias, ctx_dia_tgt_attn_bias,
                                                 params)
        if params.use_crg: # monolingual response generation task
            crg_decoder_output = transformer_decoder(decoder_input, context_dia_src_output, None,
                                                 dec_attn_bias, ctx_dia_src_attn_bias,
                                                 params) 
        if params.use_clm:
            input_tensor = state["clm"]
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                mlp_embedding = tf.get_variable("weights",
                                                [src_vocab_size, hidden_size],
                                                initializer=initializer, trainable=False)
                input_tensor = tf.layers.dense(input_tensor, hidden_size, activation=tf.nn.tanh)
            #    input_tensor = tf.nn.dropout(input_tensor, 1 - params.clm_dropout)
                input_tensor = _layer_process(input_tensor, params.layer_preprocess)
                output_bias = tf.get_variable("output_bias", shape=[src_vocab_size], initializer=tf.zeros_initializer())
#        print("clm:", input_tensor)
#        '''
            input_tensor = tf.reshape(input_tensor, [-1, hidden_size])
            clm_logits = tf.matmul(input_tensor, mlp_embedding, transpose_b=True)
            clm_logits = tf.nn.bias_add(clm_logits, output_bias)
            length = tf.shape(state["clm"])[-2]
            clm_logits = tf.reshape(clm_logits, [-1, length, src_vocab_size])
            log_probs = tf.nn.log_softmax(clm_logits, axis=-1)

    else:
#        code.interact(local=locals())
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        dia_mask = state["dia_mask"]
        length = tf.shape(decoder_input)[-2]
        batch_size = tf.shape(decoder_input)[0]
        top_mask = tf.tile(dia_mask, [1, length])
        ex_mask = tf.expand_dims(top_mask, 1)
        ti_mask = tf.tile(ex_mask, [1, params.num_heads, 1])
        ori_mask = tf.reshape(ti_mask, [-1, params.num_heads, length, enc_length])
        part_mask = tf.cast(ori_mask, dtype)
        dia_mask = inf * part_mask
        decoder_outputs = transformer_decoder(decoder_input, context_source_output, ctx_src_img_input,
                                              dec_attn_bias, ctx_src_attn_bias,
                                              params, state=state["decoder"], dia_mask=dia_mask)
#        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
#                                              dec_attn_bias, enc_attn_bias,
#                                              params, state=state["decoder"])

        decoder_output, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)

        #logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state, "context_dia_src": context_dia_src_output, "context_dia_tgt": context_dia_tgt_output, "context_source": context_source_output, "position_dia_src": turn_dia_src_pos_emb, "position_dia_tgt": turn_dia_tgt_pos_emb, "position_ctx_src": turn_ctx_src_pos_emb, "first_layer_output": first_layer_output, "sample": sample_output, "dia_mask": state["dia_mask"], "ctx_src_img_input": state["ctx_src_img_input"]}
    # for MT
    print(decoder_output, weights)
    cluts_loss = 0.0
    coh_loss = 0.0
    clus_loss = 0.0
    if params.use_clus:
        tgt_rep = tf.reduce_sum(tgt_encoder_output * t_mask, -2) / tf.reduce_sum(t_mask, -2)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            coherence_weight = tf.get_variable("cross_lingual_coherence_weights",
                                            [2, hidden_size * 2],
                                            initializer=initializer, trainable=True)
        #tf.random.shuffle(sample_rep)
        binary_1 = tf.matmul(tf.concat([context_dia_src, tgt_rep], -1), coherence_weight, False, True)
        binary_0 = tf.matmul(tf.concat([context_dia_src, sample_rep], -1), coherence_weight, False, True)

        coh1_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_1, labels=tf.ones([tf.shape(tgt_rep)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        coh2_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_0, labels=tf.zeros([tf.shape(tgt_rep)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        clus_loss = tf.reduce_mean(coh1_ce) + tf.reduce_mean(coh2_ce)

    if params.use_cluts:
        tgt_rep = tf.reduce_sum(tgt_encoder_output * t_mask, -2) / tf.reduce_sum(t_mask, -2)
        src_rep = tf.reduce_sum(encoder_output * s_mask, -2) / tf.reduce_sum(s_mask, -2)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            coherence_weight = tf.get_variable("cross_lingual_trans_selection_weights",
                                            [2, hidden_size * 2],
                                            initializer=initializer, trainable=True)
        #tf.random.shuffle(sample_rep)
        binary_1 = tf.matmul(tf.concat([src_rep, tgt_rep], -1), coherence_weight, False, True)
        binary_0 = tf.matmul(tf.concat([src_rep, sample_rep], -1), coherence_weight, False, True)

        coh1_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_1, labels=tf.ones([tf.shape(tgt_rep)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        coh2_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_0, labels=tf.zeros([tf.shape(tgt_rep)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        cluts_loss = tf.reduce_mean(coh1_ce) + tf.reduce_mean(coh2_ce)

    if params.use_coherence:
        translated = tf.reduce_sum(tgt_encoder_output * t_mask, -2) / tf.reduce_sum(t_mask, -2)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            coherence_weight = tf.get_variable("coherence_weights",
                                            [2, hidden_size * 2],
                                            initializer=initializer, trainable=True)
        tf.random.shuffle(sample_rep)
        binary_1 = tf.matmul(tf.concat([context_dia_tgt, translated], -1), coherence_weight, False, True)
        binary_0 = tf.matmul(tf.concat([context_dia_tgt, sample_rep], -1), coherence_weight, False, True)

        coh1_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_1, labels=tf.ones([tf.shape(translated)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        coh2_ce = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary_0, labels=tf.zeros([tf.shape(translated)[0], 1]), smoothing=params.label_smoothing, normalize=True)
        coh_loss = tf.reduce_mean(coh1_ce) + tf.reduce_mean(coh2_ce)

#        code.interact(local=locals())
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
#    print(decoder_output)
    logits = tf.matmul(decoder_output, weights, False, True)
#    print(logits)
    labels = features["target"]
    #code.interact(local=locals())
    # label smoothing
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    tgt_mask = tf.cast(tgt_mask, ce.dtype)
    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if params.use_mrg:
        mrg_decoder_output = tf.reshape(mrg_decoder_output, [-1, hidden_size])
        mrg_logits = tf.matmul(mrg_decoder_output, mrg_weights, False, True)
        mrg_ce = losses.smoothed_softmax_cross_entropy_with_logits(
            logits=mrg_logits,
            labels=labels,
            smoothing=params.label_smoothing,
            normalize=True
        )
        mrg_ce = tf.reshape(mrg_ce, tf.shape(tgt_seq))

    if params.use_crg:
        crg_decoder_output = tf.reshape(crg_decoder_output, [-1, hidden_size])
        crg_logits = tf.matmul(crg_decoder_output, crg_weights, False, True)
        crg_ce = losses.smoothed_softmax_cross_entropy_with_logits(
            logits=crg_logits,
            labels=labels,
            smoothing=params.label_smoothing,
            normalize=True
        )
        crg_ce = tf.reshape(crg_ce, tf.shape(tgt_seq))

    if params.use_clm:
        label_ids = features["clm_label"] #id
        label_weights = features["clm_label_position"]
        label_weights = tf.cast(label_weights, dtype)
        one_hot_labels = tf.one_hot(label_ids, depth=src_vocab_size, dtype=tf.float32)
        per_example_loss = - tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

    if mode == "eval":
        loss = -tf.reduce_sum(ce * tgt_mask, axis=1)
        if params.use_mrg:
            loss += -tf.reduce_sum(mrg_ce * tgt_mask, axis=1)
        if params.use_crg:
            loss += -tf.reduce_sum(crg_ce * tgt_mask, axis=1)
        if params.use_clm:
            loss += -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        return loss #-tf.reduce_sum(ce * tgt_mask, axis=1)

    ce_loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    mrg_ce_loss = 0.0
    crg_ce_loss = 0.0
    clm_loss = 0.0

    if params.use_mrg:
        mrg_ce_loss = tf.reduce_sum(mrg_ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    if params.use_crg:
        crg_ce_loss = tf.reduce_sum(crg_ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    if params.use_clm:
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        clm_loss = numerator / denominator

    kl_loss = mrg_ce_loss
    #code.interact(local=locals())
    # bow loss
    avg_bow_loss = crg_ce_loss
    '''
    if params.use_bowloss:
        weights_bow = tf.get_variable("softmax_bow", [tgt_vocab_size, params.latent_dim + hidden_size],
                                  initializer=initializer)
        src_latent = tf.concat([latent_sample_ctx, tgt_rep], axis=-1)
        bow_logits = tf.matmul(src_latent, weights_bow, False, True)
        tile_bow_logits = tf.tile(tf.expand_dims(bow_logits, 1), [1, tf.shape(features["target"])[1], 1])
        bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits, labels=labels) * tgt_mask
        bow_loss = tf.reduce_sum(bow_loss, reduction_indices=1)
        avg_bow_loss  = tf.reduce_mean(bow_loss)
        #bow = losses.smoothed_sigmoid_cross_entropy_with_logits(logits=bow_logits, labels=labels)
        #bow_loss = tf.reduce_sum(bow)
        #code.interact(local=locals())
    '''
    return ce_loss, kl_loss, avg_bow_loss, cluts_loss, coh_loss, clus_loss, clm_loss


def model_graph(features, mode, params):
    encoder_output, context_dia_src, context_dia_tgt, context_source, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_ctx_src_pos_emb, first_layer_output, sample_output, dia_mask, ctx_src_img_input = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output,
        "dia_mask": dia_mask,
        "context_dia_src": context_dia_src,
        "context_dia_tgt": context_dia_tgt,
        "context_source": context_source,
        "position_dia_src": turn_dia_src_pos_emb,
        "position_dia_tgt": turn_dia_tgt_pos_emb,
        "position_ctx_src": turn_ctx_src_pos_emb,
        "first_layer_output": first_layer_output,
        "sample": sample_output,
        "ctx_src_img_input": ctx_src_img_input
    }
    output = decoding_graph(features, state, mode, params)

    return output


class Transformer(NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            custom_getter = utils.custom_getter if dtype else None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                loss = model_graph(features, "train", params)
                return loss#, kl_loss, bow_loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score, _ = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output, context_dia_src, context_dia_tgt, context_source, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_ctx_src_pos_emb, first_layer_output, sample_output, dia_mask, ctx_src_img_input = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "context_dia_src": context_dia_src,
                    "context_dia_tgt": context_dia_tgt,
                    "context_source": context_source,
                    "position_dia_src": turn_dia_src_pos_emb,
                    "position_dia_tgt": turn_dia_tgt_pos_emb,
                    "position_ctx_src": turn_ctx_src_pos_emb,
                    "first_layer_output": first_layer_output,
                    "sample": sample_output,
                    "ctx_src_img_input": ctx_src_img_input,
                    "dia_mask": dia_mask,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.attention_key_channels or params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.attention_value_channels or params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            num_units=512,
            use_bowloss=False,
            use_srcctx=True,
            mrg_alpha=0.0,
            crg_alpha=0.0,
            sp_alpha=0.0,
            coh_alpha=0.0,
            use_dialog_latent=False,
            use_language_latent=False,
            use_mtstyle_latent=False,
            use_mrg=False,
            use_crg=False,
            use_speaker=False,
            use_coherence=False,
            use_clus=False,
            use_cluts=False,
            use_clm=False,
            use_emovec=False,
            segment_embeddings=False,
            hidden_size=512,
            latent_dim=32,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.5,
            relu_dropout=0.0,
            clm_dropout=0.1,
            embed_dropout=0.3,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            context_representation="self_attention",
            num_context_layers=1,
            bottom_block=6,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            # "absolute" or "relative"
            position_info_type="relative",
            # 8 for big model, 16 for base model, see (Shaw et al., 2018)
            max_relative_dis=16
        )

        return params
