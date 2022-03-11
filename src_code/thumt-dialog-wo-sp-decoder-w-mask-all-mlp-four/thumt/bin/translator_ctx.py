#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os,code
import six
import numpy as np
import tensorflow as tf
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.inference_ctx as inference
import thumt.utils.parallel as parallel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")

    parser.add_argument("--dev_dialog_src_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_dialog_tgt_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_style_src_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_style_tgt_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_language_src_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_language_tgt_context", type=str,
                        help="Path of dev_context corpus")

    parser.add_argument("--emotion", type=str,
                        help="Path to emotion file")
    parser.add_argument("--dev_emotion", type=str,
                        help="Path to dev_emotion file")
    # model and configuration
    parser.add_argument("--models", type=str, required=True, nargs="+",
                        help="Name of the model")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        embedding_path="/mnt/yardcephfs/mmyard/g_wxg_td_prc/yunlonliang/4.sa/embedding/glove.840B.300d.emotion.txt",
        start_steps=0,
        kl_annealing_steps=10000,
        append_eos=False,
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        decode_batch_size=32,
        device_list=[0],
        num_threads=1
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

#    for (k, v) in params1.values().iteritems():
    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

#    for (k, v) in params2.values().iteritems():
    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    if model_name.startswith("experimental_"):
        model_name = model_name[13:]

    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    if args.parameters:
        params.parse(args.parameters)

    params.dev_dialog_src_context = args.dev_dialog_src_context #or params.dev_dialog_src_context
    params.dev_dialog_tgt_context = args.dev_dialog_tgt_context #or params.dev_dialog_tgt_context
    params.dev_style_src_context = args.dev_style_src_context #or params.dev_style_src_context
    params.dev_style_tgt_context = args.dev_style_tgt_context #or params.dev_style_tgt_context
    params.dev_language_src_context = args.dev_language_src_context #or params.dev_language_src_context
    params.dev_language_tgt_context = args.dev_language_tgt_context #or params.dev_language_tgt_context
    params.dev_emotion = args.dev_emotion or params.dev_emotion
    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary[0]),
        "target": vocabulary.load_vocabulary(args.vocabulary[1]),
        "emotion": ['neutral', 'joy', 'anger', 'surprise', 'sadness', 'fear', 'disgust', 'happiness', 'happy', 'other'],
        "position": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }

    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break

    return ops


def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in features:
        feat = features[name]
        batch = feat.shape[0]

        if batch < num_shards:
            feed_dict[placeholders[0][name]] = feat
            n = 1
        else:
            shard_size = (batch + num_shards - 1) // num_shards

            for i in range(num_shards):
                shard_feat = feat[i * shard_size:(i + 1) * shard_size]
                feed_dict[placeholders[i][name]] = shard_feat
                n = num_shards

    return predictions[:n], feed_dict

def get_turn_position(file1):
    with open(file1, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
    turn_position = []
    for line in content:
        tmp = []
        index = 1
        for i in line.split()[::-1]:
            #tmp.append(str(index))
            if i == '[SEP]':
                index += 1
            tmp.append(str(index))
        turn_position.append(tmp)
    base_path = '/'.join(file1.split('/')[:-1])
    signal = file1.split('/')[-1] #.split('.')[0]
    position_file = base_path + '/' + signal + '.turn_position'
    with open(position_file, 'w', encoding='utf-8') as fw:
        for line_position in turn_position:
            line_position = sorted(line_position, reverse=True)
            fw.write(' '.join(line_position) + '\n')
    #code.interact(local=locals())
    return position_file

def get_turn_position_src(file1, file2):
    with open(file1, 'r', encoding='utf-8') as fr:
        sentence = fr.readlines()
    with open(file2, 'r', encoding='utf-8') as fr:
        content = fr.readlines()

    turn_position = []
    mask = []
    dialog = []
    for src, ctx in zip(sentence, content):
        line = ctx.replace('\n', ' ') + src.replace('\n', ' ') + '<eos> '
        dialog.append(line)
        tmp = []
        mask_sent = []
        index = 1
        flag = 0
        for i in line.split()[::-1]:
            if flag == 0:
                mask_sent.append(str(1))
                tmp.append(str(0))
            else:
                #mask_sent.append(str(0))
                tmp.append(str(index))
                if i == '[SEP]':
                    index += 1

            if i == '<eos>':
                flag = 1

        if len(line.split()) != len(tmp):
            print(line)
        turn_position.append(tmp)
        mask.append(mask_sent)

    base_path = '/'.join(file1.split('/')[:-1])
    signal = file1.split('/')[-1] #.split('.')[0]
    position_file = base_path + '/' + signal + '.src_dia_turn_position'
    dialog_file = base_path + '/' + signal + '.src.dialog'

    with open(position_file, 'w', encoding='utf-8') as fw:
        for line_position in turn_position:
            line_position = sorted(line_position, reverse=True)
            fw.write(' '.join(line_position) + '\n')

    with open(dialog_file, 'w', encoding='utf-8') as fw:
        for sub_mask in dialog:
            fw.write(sub_mask)

    #code.interact(local=locals())
    return position_file, dialog_file

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    model_cls_list = [models.get_model(model) for model in args.models]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.checkpoints[i], args.models[i], params_list[i])
        for i in range(len(args.checkpoints))
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]

    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        for i, checkpoint in enumerate(args.checkpoints):
            tf.logging.info("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                if not name.startswith(model_cls_list[i].get_name()):
                    continue

                if name.find("losses_avg") >= 0:
                    continue

                tensor = reader.get_tensor(name)
                values[name] = tensor

            model_var_lists.append(values)

        # Build models
        model_fns = []

        for i in range(len(args.checkpoints)):
            name = model_cls_list[i].get_name()
            model = model_cls_list[i](params_list[i], name + "_%d" % i)
            model_fn = model.get_inference_func()
            model_fns.append(model_fn)

        params = params_list[0]
        # Read input file
        position_file_src_dia = get_turn_position(params.dev_dialog_src_context)
#            position_file_src_dia, dialog_file = get_turn_position_src(params.validation, params.dialog_src_context)

        position_file_tgt_dia = get_turn_position(params.dev_dialog_tgt_context)
        position_file_src_sty = get_turn_position(params.dev_style_src_context)
        position_file_tgt_sty = get_turn_position(params.dev_style_tgt_context)
        position_file_src_lan = get_turn_position(params.dev_language_src_context)
        position_file_tgt_lan = get_turn_position(params.dev_language_tgt_context)

        sorted_keys, sorted_inputs, dialog_src_context, pos_src_dia, dialog_tgt_context, pos_tgt_dia, style_src_context, pos_src_sty, style_tgt_context, pos_tgt_sty, language_src_context, pos_src_lan, language_tgt_context, pos_tgt_lan, emo = dataset.sort_input_file_ctx(args.input, params.dev_dialog_src_context, position_file_src_dia, params.dev_dialog_tgt_context, position_file_tgt_dia, params.dev_style_src_context, position_file_src_sty, params.dev_style_tgt_context, position_file_tgt_sty, params.dev_language_src_context, position_file_src_lan, params.dev_language_tgt_context, position_file_tgt_lan, params.dev_emotion)
        # Build input queue
        features = dataset.get_inference_input(sorted_inputs, "x", params)
#        features_ctx = dataset.get_inference_input(sorted_ctxs, params)

        features_ctx1 = dataset.get_inference_input(dialog_src_context, "ctx",  params)
        features_ctx2 = dataset.get_inference_input(pos_src_dia, "pos", params)
        features_ctx3 = dataset.get_inference_input(dialog_tgt_context, "ctx",  params)
        features_ctx4 = dataset.get_inference_input(pos_tgt_dia, "pos", params)
        features_ctx5 = dataset.get_inference_input(style_src_context, "ctx",  params)
        features_ctx6 = dataset.get_inference_input(pos_src_sty, "pos", params)
        features_ctx7 = dataset.get_inference_input(style_tgt_context, "ctx",  params)
        features_ctx8 = dataset.get_inference_input(pos_tgt_sty, "pos", params)
        features_ctx9 = dataset.get_inference_input(language_src_context, "ctx", params)
        features_ctx10 = dataset.get_inference_input(pos_src_lan, "pos", params)
        features_ctx11 = dataset.get_inference_input(language_tgt_context, "ctx", params)
        features_ctx12 = dataset.get_inference_input(pos_tgt_lan, "pos", params)
        features_ctx13 = dataset.get_inference_input(emo, "emo", params)

#        features["context"] = features_ctx["source"]
#        features["context_length"] = features_ctx["source_length"]

        features["context_dia_src"] = features_ctx1["source"]
        features["context_dia_src_length"] = features_ctx1["source_length"]
        features["position_dia_src"] = features_ctx2["source"]
#        features[""] = features_ctx2["source_length"]
        features["context_dia_tgt"] = features_ctx3["source"]
        features["context_dia_tgt_length"] = features_ctx3["source_length"]
        features["position_dia_tgt"] = features_ctx4["source"]
#       features[""] = features_ctx4["source_length"]
        features["context_sty_src"] = features_ctx5["source"]
        features["context_sty_src_length"] = features_ctx5["source_length"]
        features["position_sty_src"] = features_ctx6["source"]
#        features[""] = features_ctx6["source_length"]
        features["context_sty_tgt"] = features_ctx7["source"]
        features["context_sty_tgt_length"] = features_ctx7["source_length"]
        features["position_sty_tgt"] = features_ctx8["source"]
#        features[""] = features_ctx8["source_length"]
        features["context_lan_src"] = features_ctx9["source"]
        features["context_lan_src_length"] = features_ctx9["source_length"]
        features["position_lan_src"] = features_ctx10["source"]
#        features[""] = features_ctx10["source_length"]
        features["context_lan_tgt"] = features_ctx11["source"]
        features["context_lan_tgt_length"] = features_ctx11["source_length"]
        features["position_lan_tgt"] = features_ctx12["source"]
#        features[""] = features_ctx12["source_length"]
        features["emotion"] = features_ctx13["source"]
#        features[""] = features_ctx13["source_length"]
        # Create placeholders
        placeholders = []

        for i in range(len(params.device_list)):
            placeholders.append({
                "source": tf.placeholder(tf.int32, [None, None],
                                         "source_%d" % i),
                "source_length": tf.placeholder(tf.int32, [None],
                                                "source_length_%d" % i),
                "context_dia_src": tf.placeholder(tf.int32, [None, None], "context_dia_src_%d" % i),
                "context_dia_src_length": tf.placeholder(tf.int32, [None], "context_dia_src_length_%d" % i),
                "context_dia_tgt": tf.placeholder(tf.int32, [None, None], "context_dia_tgt_%d" % i),
                "context_dia_tgt_length": tf.placeholder(tf.int32, [None], "context_dia_tgt_length_%d" % i),
                "context_sty_src": tf.placeholder(tf.int32, [None, None], "context_sty_src_%d" % i),
                "context_sty_src_length": tf.placeholder(tf.int32, [None], "context_sty_src_length_%d" % i),
                "context_sty_tgt": tf.placeholder(tf.int32, [None, None], "context_sty_tgt_%d" % i),
                "context_sty_tgt_length": tf.placeholder(tf.int32, [None], "context_sty_tgt_length_%d" % i),
                "context_lan_src": tf.placeholder(tf.int32, [None, None], "context_lan_src_%d" % i),
                "context_lan_src_length": tf.placeholder(tf.int32, [None], "context_lan_src_length_%d" % i),
                "context_lan_tgt": tf.placeholder(tf.int32, [None, None], "context_lan_tgt_%d" % i),
                "context_lan_tgt_length": tf.placeholder(tf.int32, [None], "context_lan_tgt_length_%d" % i),
                "emotion": tf.placeholder(tf.int32, [None, None], "emotion_%d" % i),
                "position_dia_src": tf.placeholder(tf.int32, [None, None], "position_dia_src_%d" % i),
                "position_dia_tgt": tf.placeholder(tf.int32, [None, None], "position_dia_tgt_%d" % i),
                "position_sty_src": tf.placeholder(tf.int32, [None, None], "position_sty_src_%d" % i),
                "position_sty_tgt": tf.placeholder(tf.int32, [None, None], "position_sty_tgt_%d" % i),
                "position_lan_src": tf.placeholder(tf.int32, [None, None], "position_lan_src_%d" % i),
                "position_lan_tgt": tf.placeholder(tf.int32, [None, None], "position_lan_tgt_%d" % i)
            })

        # A list of outputs
        predictions = parallel.data_parallelism(
            params.device_list,
            lambda f: inference.create_inference_graph(model_fns, f, params),
            placeholders)

        # Create assign ops
        assign_ops = []

        all_var_list = tf.all_variables()

        for i in range(len(args.checkpoints)):
            un_init_var_list = []
            name = model_cls_list[i].get_name()

            for v in all_var_list:
                if v.name.startswith(name + "_%d" % i):
                    un_init_var_list.append(v)

            ops = set_variables(un_init_var_list, model_var_lists[i],
                                name + "_%d" % i)
            assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)
        results = []

        # Create session
        with tf.Session(config=session_config(params)) as sess:
            # Restore variables
            sess.run(assign_op)
            sess.run(tf.tables_initializer())

            while True:
                try:
                    feats = sess.run(features)
                    op, feed_dict = shard_features(feats, placeholders,
                                                   predictions)
                    res = sess.run(predictions, feed_dict=feed_dict)
                    results.append(res)
                    print("res:", res)
                    message = "Finished batch %d" % len(results)
                    tf.logging.log(tf.logging.INFO, message)
                except tf.errors.OutOfRangeError:
                    break

        # Convert to plain text
        vocab = params.vocabulary["target"]
        outputs = []
        scores = []

        for result in results:
            for item in result[0]:
                outputs.append(item.tolist())
            for item in result[1]:
                scores.append(item.tolist())

        outputs = list(itertools.chain(*outputs))
        scores = list(itertools.chain(*scores))

        restored_inputs = []
        restored_outputs = []
        restored_scores = []

        for index in range(len(sorted_inputs)):
            restored_inputs.append(sorted_inputs[sorted_keys[index]])
            restored_outputs.append(outputs[sorted_keys[index]])
            restored_scores.append(scores[sorted_keys[index]])

        # Write to file
        with open(args.output, "w") as outfile:
            count = 0
            for outputs, scores in zip(restored_outputs, restored_scores):
                for output, score in zip(outputs, scores):
                    decoded = []
                    for idx in output:
                        if idx == params.mapping["target"][params.eos]:
                            break
                        decoded.append(vocab[idx])

                    decoded = " ".join(decoded)

                    if not args.verbose:
                        outfile.write("%s\n" % decoded)
                        break
                    else:
                        pattern = "%d ||| %s ||| %s ||| %f\n"
                        source = restored_inputs[count]
                        values = (count, source, decoded, score)
                        outfile.write(pattern % values)

                count += 1

if __name__ == "__main__":
    main(parse_args())
