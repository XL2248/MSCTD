# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import code
import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers
from tensorflow.contrib import rnn


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

def _residual_gating_fn(x, y, hidden_size, keep_prob=None):
    with tf.variable_scope("gating_x"):
        gating_x = layers.nn.linear(x, hidden_size, False)
    with tf.variable_scope("gating_y"):
        gating_y = layers.nn.linear(y, hidden_size, False)
    gate = tf.sigmoid(gating_x+gating_y) 
    #if keep_prob and keep_prob < 1.0:
    #    y = tf.nn.dropout(y, keep_prob)
    return gate*x+(1-gate)*y

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

def w_encoder_attention(queries,
                        keys,
                        sequence_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        using_mask=False,
                        mymasks=None,
                        scope="w_encoder_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Set the fall back option for num_units
        # print(queries)
        # print(queries.get_shape().as_list)
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        # Linear projections

        Q = tf.layers.dense(queries, num_units, activation=None, use_bias=False)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None, use_bias=False)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None, use_bias=False)  # (N, T_k, C)

        x = K * Q
        x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1],num_heads, int(num_units/num_heads)])
        outputs = tf.transpose(tf.reduce_sum(x, 3),[0,2,1])
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)

        if using_mask:
            key_masks = mymasks
            key_masks = tf.reshape(tf.tile(key_masks, [1, num_heads]),
                                   [tf.shape(key_masks)[0], num_heads, tf.shape(key_masks)[1]])
        else:
            key_masks = tf.sequence_mask(sequence_length, tf.shape(keys)[1], dtype=tf.float32)
            key_masks = tf.reshape(tf.tile(key_masks,[1, num_heads]),[tf.shape(key_masks)[0],num_heads,tf.shape(key_masks)[1]])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        outputs = tf.nn.softmax(outputs, 2)
        V_ = tf.reshape(V, [tf.shape(V)[0], tf.shape(V)[1], num_heads, int(num_units / num_heads)])
        V_ = tf.transpose(V_, [0, 2, 1, 3])
        outputs = tf.layers.dense(tf.reshape(tf.reduce_sum(V_ * tf.expand_dims(outputs, -1), 2), [-1, num_units]),
                                  num_units, activation=None, use_bias=False)
        weight = outputs
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    return outputs, weight

def transformer_context(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="context_transformer", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_context_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("ctx_self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        trainable=True
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
                        trainable=True
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs

def transformer_encoder(inputs, memory_ctx, bias, bias_ctx, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
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
                        trainable=params.is_training
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=params.is_training)

                if params.context_encoder_attention:
                    with tf.variable_scope("ctxenc_attention"):
                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            memory_ctx,
                            bias_ctx,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            trainable=True
                        )
                        y = y["outputs"]

                        if params.context_gating:
                            x = _residual_gating_fn(x, y, params.hidden_size, 1.0 - params.residual_dropout)
                            x = _layer_process(x, params.layer_postprocess)
                        else:
                            x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                            x = _layer_process(x, params.layer_postprocess)



                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        trainable=params.is_training
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=params.is_training)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, memory_ctx, bias, mem_bias, bias_ctx, 
                        params, state=None, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

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
                        trainable=params.is_training
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=params.is_training)

                if params.context_decoder_attention:
                    with tf.variable_scope("ctxdec_attention"):
                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            memory_ctx,
                            bias_ctx,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            trainable=True
                        )
                        y = y["outputs"]
                        if params.context_gating:
                            x = _residual_gating_fn(x, y, params.hidden_size, 1.0 - params.residual_dropout)
                            x = _layer_process(x, params.layer_postprocess)
                        else:
                            x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                            x = _layer_process(x, params.layer_postprocess)

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
                        trainable=params.is_training
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=params.is_training)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        trainable=params.is_training
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=params.is_training)

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

    hidden_size = params.hidden_size
    src_seq = features["source"]
    ctx_seq = features["context"]
    src_len = features["source_length"]
    ctx_len = features["context_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    ctx_mask = tf.sequence_mask(ctx_len,
                                maxlen=tf.shape(features["context"])[1],
                                dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer, trainable=False)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer, trainable=False)

    bias = tf.get_variable("bias", [hidden_size], trainable=False)

    ## context
    # ctx_seq: [batch, max_ctx_length]
    print("building context graph")
    if params.context_representation == "self_attention":
        print('use self attention')
        ctx_inputs = tf.gather(src_embedding, ctx_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        ctx_attn_bias = layers.attention.attention_bias(ctx_mask, "masking")

        context_output = transformer_context(context_input, ctx_attn_bias, params)
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


    ## encoder

    # id => embedding
    # src_seq: [batch, max_src_length]
    print("building encoder graph")
    inputs = tf.gather(src_embedding, src_seq) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(src_mask, -1)

    # Preparing encoder
    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = layers.attention.add_timing_signal(encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    if params.context_encoder_attention:
        encoder_output = transformer_encoder(encoder_input, context_output, enc_attn_bias, ctx_attn_bias, params)
    else:
        encoder_output = transformer_encoder(encoder_input, None, enc_attn_bias, None, params)

    return context_output, encoder_output


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0
        is_training = False
    else:
        is_training = True

    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    ctx_len = features["context_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)
    ctx_mask = tf.sequence_mask(ctx_len,
                                maxlen=tf.shape(features["context"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer, trainable=False)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer, trainable=False)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer, trainable=False)

    # id => embedding
    # tgt_seq: [batch, max_tgt_length]
    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)
    targets = targets * tf.expand_dims(tgt_mask, -1)

    # Preparing encoder and decoder input
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    ctx_attn_bias = layers.attention.attention_bias(ctx_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]
    context_output = state["context"]
    latent_sample = []
    w_query = tf.get_variable("w_Q", [1, params.num_units], dtype=tf.float32)

    # cvae
#    '''
    context_rep = tf.reduce_mean(context_output, -2)
    context_rep = tf.tile(tf.expand_dims(context_rep, 1), [1, tf.shape(encoder_output)[-2], 1])
    encoder_output1 = tf.concat([encoder_output, context_rep], axis=-1)

    prior_encode, w_ = w_encoder_attention(w_query,
                                          tf.layers.dense(encoder_output1, 512, activation=tf.nn.tanh),
                                          src_len,
                                          num_units=params.num_units,
                                          num_heads=params.num_heads,
                                          dropout_rate=params.residual_dropout,
                                          is_training=is_training,
                                          mymasks=src_mask,
                                          scope="concentrate_attention",
                                          reuse=tf.AUTO_REUSE
                                          )
    '''
    #post_encode = tf.reduce_mean(decoder_output, -2)

    encoder_rep = tf.reduce_mean(encoder_output, -2)
    context_rep = tf.reduce_mean(context_output, -2)
    prior_encode = tf.concat([encoder_rep, context_rep], axis=-1)
    '''
    prior_mulogvar = tf.layers.dense(tf.layers.dense(prior_encode, 256, activation=tf.nn.tanh), params.latent_dim * 2, use_bias=False, name="prior_fc")
    prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)

    if mode != "infer":
        if params.context_decoder_attention:
            decoder_output = transformer_decoder(decoder_input, encoder_output, context_output,
                                             dec_attn_bias, enc_attn_bias, ctx_attn_bias,
                                             params)
        else:
            decoder_output = transformer_decoder(decoder_input, encoder_output, None,
                                             dec_attn_bias, enc_attn_bias, None,
                                             params)
#        '''
        post_encode, w_ = w_encoder_attention(w_query,
                                             decoder_output,
                                             tgt_len,
                                             num_units=params.num_units,
                                             num_heads=params.num_heads,
                                             dropout_rate=params.residual_dropout,
                                             is_training=is_training,
                                             using_mask=False,
                                             mymasks=None,
                                             scope="concentrate_attention"
                                             )
#        '''
#        post_encode = tf.reduce_mean(decoder_output, -2)
        post_mulogvar = tf.layers.dense(post_encode, params.latent_dim * 2, use_bias=False, name="post_fc")
        post_mu, post_logvar = tf.split(post_mulogvar, 2, axis=1)

        latent_sample = sample_gaussian(post_mu, post_logvar)
        #code.interact(local=locals())

    else:
        latent_sample = sample_gaussian(prior_mu, prior_logvar)
        #latent_sample = tf.tile(tf.expand_dims(latent_sample, 1), [1, params.decode_length, 1])
#        inputs = tf.concat([inputs, latent_sample], axis=2)
#        inputs = tf.layers.dense(inputs, params.num_units, activation=tf.tanh, use_bias=False, name="last")

        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output, context_output,
                                              dec_attn_bias, enc_attn_bias, ctx_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
#        code.interact(local=locals())
        out_lat = tf.concat([decoder_output, latent_sample], axis=-1)
        out_lat1 = tf.layers.dense(out_lat, params.num_units, activation=tf.tanh, use_bias=False, name="last")
        logits = tf.matmul(out_lat1, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state, "context": context_output}

    # [batch, length, channel] => [batch * length, vocab_size]
#    decoder_output1 = tf.reshape(decoder_output, [-1, hidden_size])
    # cvae
#    length = decoder_output.get_shape().as_list()[-2]
#    code.interact(local=locals())
#    latent_sample = tf.tile(tf.expand_dims(latent_sample, 1), [1, length, 1])
    latent_sample = tf.tile(tf.expand_dims(latent_sample, 1), [1, tf.shape(decoder_output)[-2], 1])
    #code.interact(local=locals())
    out_lat = tf.concat([decoder_output, latent_sample], axis=-1)
    out_lat1 = tf.layers.dense(out_lat, hidden_size, activation=tf.tanh, use_bias=False, name="last")
    decoder_output1 = tf.reshape(out_lat1, [-1, hidden_size])
    logits = tf.matmul(decoder_output1, weights, False, True)
    #logits = tf.matmul(out_lat, weights, False, True)
    labels = features["target"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
#    code.interact(local=locals())
    #kl loss
    kl_weights = tf.minimum(tf.to_float(params.global_step) / 10000, 1.0)
    kld = gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar)
    kl_loss = tf.reduce_mean(kld) * kl_weights
    print('global, kl_loss',params.global_step, kl_loss)

    return 1.0*loss + 1.0*kl_loss


def model_graph(features, mode, params):
    context_output, encoder_output = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output,
        "context": context_output
    }
    output = decoding_graph(features, state, mode, params)

    return output


class Contextual_Transformer(interface.NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Contextual_Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                context_output, encoder_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "context": context_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.hidden_size])
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
            is_training=False,
            append_eos=False,
            decode_length=50,
            latent_dim=16,
            num_units=512,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # contextual 
            context_encoder_attention=False,
            context_decoder_attention=False,
            context_gating=False,
            context_representation="self_attention",
            num_context_layers=6,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=0.1,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0
        )

        return params
