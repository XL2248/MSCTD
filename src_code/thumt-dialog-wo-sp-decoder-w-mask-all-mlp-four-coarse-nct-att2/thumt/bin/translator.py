#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os,code
import six
import sys

import numpy as np
import tensorflow as tf
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.inference as inference
import thumt.utils.parallel as parallel
import thumt.utils.sampling as sampling


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
    parser.add_argument("--valid_index", type=str,
                        help="Path of validation file")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=4, required=True,
                        help="Path of source and target vocabulary")

    parser.add_argument("--dev_dialog_src_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_dialog_tgt_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_sample", type=str,
                        help="Path of sample corpus")

    parser.add_argument("--dev_context_source", type=str,
                        help="Path of sample corpus")

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
        append_eos=False,
        device_list=[0],
        num_threads=1,
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        start_steps=0,
        kl_annealing_steps=10000,
        decode_batch_size=32,
        # sampling
        generate_samples=False,
        num_samples=1,
        min_length_ratio=0.0,
        max_length_ratio=1.5,
        min_sample_length=0,
        max_sample_length=0,
        sample_batch_size=32
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

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


    params.dev_context_source = args.dev_context_source

    params.dev_dialog_src_context = args.dev_dialog_src_context #or params.dev_dialog_src_context
    params.dev_dialog_tgt_context = args.dev_dialog_tgt_context #or params.dev_dialog_tgt_context
    params.valid_index = args.valid_index
    params.dev_sample = args.dev_sample

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary[0]),
        "target": vocabulary.load_vocabulary(args.vocabulary[1]),
        "position": vocabulary.load_vocabulary(args.vocabulary[2]),
        "index": vocabulary.load_vocabulary(args.vocabulary[3])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )
    params.vocabulary["index"] = vocabulary.process_vocabulary(
        params.vocabulary["index"], params
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
        ),
        "index": vocabulary.get_control_mapping(
            params.vocabulary["index"],
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


def set_variables(var_list, value_dict, prefix, feed_dict):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                placeholder = tf.placeholder(tf.float32,
                                             name="placeholder/" + var_name)
                with tf.device("/cpu:0"):
                    op = tf.assign(var, placeholder)
                    ops.append(op)
                feed_dict[placeholder] = value_dict[name]
                break

    return ops

def get_turn_index(src, ctx_src, src_ctx, tgt_ctx, index):

    with open(src, 'r', encoding='utf-8') as fr:
        content1 = fr.readlines()
    with open(ctx_src, 'r', encoding='utf-8') as fr:
        content2 = fr.readlines()
    with open(src_ctx, 'r', encoding='utf-8') as fr:
        content3 = fr.readlines()
    with open(tgt_ctx, 'r', encoding='utf-8') as fr:
        content4 = fr.readlines()
    with open(index, 'r', encoding='utf-8') as fr:
        content5 = fr.readlines()

    src_idx, ctx_src_idx, src_ctx_idx, tgt_ctx_idx = [], [], [], []
    for idx in range(len(content1)):
        tmp = []
        line1 = content1[idx].strip() + " <eos>"
        line2 = content2[idx].strip()
        line3 = content3[idx].strip()
        line4 = content4[idx].strip()
        indx = content5[idx].strip()
        for i in line1.strip().split(' ')[::-1]:
            tmp.append(indx)
        src_idx.append(tmp)

        tmp = []
        number = int(indx) 
        for i in line2.strip().split(' ')[::-1]:
            tmp.append(str(number))
            if i == '[SEP]':
                number -= 1
        ctx_src_idx.append(tmp)

        tmp = []
        number = int(indx) + 1
        for i in line3.strip().split(' ')[::-1]:
            tmp.append(str(number -1))
            if i == '[SEP]':
                number -= 1
        src_ctx_idx.append(tmp)

        tmp = []
        number = int(indx) + 1
        for i in line4.strip().split(' ')[::-1]:
            tmp.append(str(number -1))
            if i == '[SEP]':
                number -= 1
        tgt_ctx_idx.append(tmp)

    base_path = '/'.join(src.split('/')[:-1])
    signal = src.split('/')[-1] #.split('.')[0]
    src_file = base_path + '/' + signal + '.src_index'
    ctx_src_file = base_path + '/' + signal + '.ctx_src_index'
    src_ctx_file = base_path + '/' + signal + '.src_ctx_index'
    tgt_ctx_file = base_path + '/' + signal + '.tgt_ctx_index'

#    if os.path.exists(position_file) and os.path.exists(mask_file):
    if os.path.exists(src_file) and not os.path.getsize(src_file):
        return src_file, ctx_src_file, src_ctx_file, tgt_ctx_file

    with open(src_file, 'w', encoding='utf-8') as fw:
        for line_position in src_idx:
            fw.write(' '.join(line_position) + '\n')

    with open(ctx_src_file, 'w', encoding='utf-8') as fw:
        for line_mask in ctx_src_idx:
            line_mask = sorted(line_mask, reverse=False)
            fw.write(' '.join(line_mask) + '\n')
#    code.interact(local=locals())
    with open(src_ctx_file, 'w', encoding='utf-8') as fw:
        for line_mask in src_ctx_idx:
            tmp = [line_mask[0]]
            temp = sorted(line_mask, reverse=False)
            for i in range(len(line_mask) -1):
                tmp.append(temp[i])
            fw.write(' '.join(tmp) + '\n')

    with open(tgt_ctx_file, 'w', encoding='utf-8') as fw:
        for line_mask in tgt_ctx_idx:
            tmp = [line_mask[0]]
            temp = sorted(line_mask, reverse=False)
            for i in range(len(line_mask) -1):
                tmp.append(temp[i])
            fw.write(' '.join(tmp) + '\n')

    return src_file, ctx_src_file, src_ctx_file, tgt_ctx_file

def get_turn_position_eos(file1):
    with open(file1, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
    turn_position = []
    mask = []
    for line in content:
        tmp = []
        mask_tmp = []
        index = 0
        lines = line.strip() + " <eos>"
        flag = 0
        for i in lines.strip().split(' ')[::-1]:
            #tmp.append(str(index))
#            flag = 0
            if i == '[SEP]':
                index += 1
                flag = 1
            tmp.append(str(index))
            mask_tmp.append(str(flag))
#        if len(lines.split()) != len(tmp):
        if len(lines.strip().split(' ')) != len(tmp):
            print(line)
        turn_position.append(tmp)
        mask.append(mask_tmp)

    base_path = '/'.join(file1.split('/')[:-1])
    signal = file1.split('/')[-1] #.split('.')[0]
    position_file = base_path + '/' + signal + '.turn_position'
    mask_file = base_path + '/' + signal + '.mask'

    if os.path.exists(position_file) and os.path.exists(mask_file) and not os.path.getsize(position_file) and not os.path.getsize(mask_file):
        return position_file, mask_file

    with open(position_file, 'w', encoding='utf-8') as fw:
        for line_position in turn_position:
            line_position = sorted(line_position, reverse=True)
            fw.write(' '.join(line_position) + '\n')
    with open(mask_file, 'w', encoding='utf-8') as fw:
        for line_mask in mask:
            line_mask = sorted(line_mask, reverse=True)
            fw.write(' '.join(line_mask) + '\n')
    #code.interact(local=locals())
    return position_file, mask_file

def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in features:
        feat = features[name]
        batch = feat.shape[0]
        shard_size = (batch + num_shards - 1) // num_shards

        for i in range(num_shards):
            shard_feat = feat[i * shard_size:(i + 1) * shard_size]

            if shard_feat.shape[0] != 0:
                feed_dict[placeholders[i][name]] = shard_feat
                n = i + 1
            else:
                break

    if isinstance(predictions, (list, tuple)):
        predictions = predictions[:n]

    return predictions, feed_dict
def get_turn_position(file1):
    with open(file1, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
    turn_position = []
    for line in content:
        tmp = []
        index = 0
        for i in line.split()[::-1]:
            #tmp.append(str(index))
            if i == '[SEP]':
                index += 1
            tmp.append(str(index))
        turn_position.append(tmp)
    base_path = '/'.join(file1.split('/')[:-1])
    signal = file1.split('/')[-1] #.split('.')[0]
    position_file = base_path + '/' + signal + '.turn_position'
    if os.path.exists(position_file) and not os.path.getsize(position_file):
        return position_file
    with open(position_file, 'w', encoding='utf-8') as fw:
        for line_position in turn_position:
            line_position = sorted(line_position, reverse=True)
            fw.write(' '.join(line_position) + '\n')
    #code.interact(local=locals())
    return position_file

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
        model_list = []

        for i in range(len(args.checkpoints)):
            name = model_cls_list[i].get_name()
            model = model_cls_list[i](params_list[i], name + "_%d" % i)
            model_list.append(model)

        params = params_list[0]
        # Read input file
        # Build input queue
#        features = dataset.get_inference_input(sorted_inputs, params)
        position_file_src_dia = get_turn_position(params.dev_dialog_src_context)
#            position_file_src_dia, dialog_file = get_turn_position_src(params.validation, params.dialog_src_context)
        position_file_tgt_dia = get_turn_position(params.dev_dialog_tgt_context)
#            position_file_ctx_src, mask_file = get_turn_position_eos(params.dev_context_source)
        position_file_ctx_src = get_turn_position(params.dev_context_source)

        src_img_idx, ctx_src_img_idx, src_ctx_img_idx, tgt_ctx_img_idx = get_turn_index(args.input, params.dev_context_source, params.dev_dialog_src_context, params.dev_dialog_tgt_context, params.valid_index)

        sorted_keys, sorted_inputs, dialog_src_context, pos_src_dia, dialog_tgt_context, pos_tgt_dia, dialog_context_source, pos_ctx_src, img1, img2, img3, img4 = dataset.sort_input_file_ctx(args.input, params.dev_dialog_src_context, position_file_src_dia, params.dev_dialog_tgt_context, position_file_tgt_dia, params.dev_context_source, position_file_ctx_src, src_img_idx, ctx_src_img_idx, src_ctx_img_idx, tgt_ctx_img_idx)
        # Build input queue
        features = dataset.get_inference_input(sorted_inputs, "x", params)
#        features_ctx = dataset.get_inference_input(sorted_ctxs, params)

        features_ctx1 = dataset.get_inference_input(dialog_src_context, "ctx",  params)
        features_ctx2 = dataset.get_inference_input(pos_src_dia, "pos", params)
        features_ctx3 = dataset.get_inference_input(dialog_tgt_context, "ctx",  params)
        features_ctx4 = dataset.get_inference_input(pos_tgt_dia, "pos", params)
        features_ctx5 = dataset.get_inference_input(dialog_context_source, "ctx",  params)
        features_ctx6 = dataset.get_inference_input(pos_ctx_src, "pos", params)
        features_ctx7 = dataset.get_inference_input(img1, "img", params)
        features_ctx8 = dataset.get_inference_input(img2, "img", params)
        features_ctx9 = dataset.get_inference_input(img3, "img", params)
        features_ctx10 = dataset.get_inference_input(img4, "img", params)
        # Create placeholders
        placeholders = []

        features["context_dia_src"] = features_ctx1["source"]
        features["context_dia_src_length"] = features_ctx1["source_length"]
        features["position_dia_src"] = features_ctx2["source"]
#        features[""] = features_ctx2["source_length"]
        features["context_dia_tgt"] = features_ctx3["source"]
        features["context_dia_tgt_length"] = features_ctx3["source_length"]
        features["position_dia_tgt"] = features_ctx4["source"]
#       features[""] = features_ctx4["source_length"]
        features["context_source"] = features_ctx5["source"]
        features["context_source_length"] = features_ctx5["source_length"]
        features["position_ctx_src"] = features_ctx6["source"]
        features["src_image"] = features_ctx7["source"]
        features["ctx_src_image"] = features_ctx8["source"]
        features["src_ctx_image"] = features_ctx9["source"]
        features["tgt_ctx_image"] = features_ctx10["source"]
        features["sample"] = features["source"]
        features["sample_length"] = features["source_length"]

        for i in range(len(params.device_list)):
            placeholders.append({
                "source": tf.placeholder(tf.int32, [None, None],
                                         "source_%d" % i),
                "source_length": tf.placeholder(tf.int32, [None],
                                                "source_length_%d" % i),
                "sample": tf.placeholder(tf.int32, [None, None],
                                         "sample_%d" % i),
                "sample_length": tf.placeholder(tf.int32, [None],
                                                "sample_length_%d" % i),
                "context_dia_src": tf.placeholder(tf.int32, [None, None], "context_dia_src_%d" % i),
                "context_dia_src_length": tf.placeholder(tf.int32, [None], "context_dia_src_length_%d" % i),
                "context_dia_tgt": tf.placeholder(tf.int32, [None, None], "context_dia_tgt_%d" % i),
                "context_dia_tgt_length": tf.placeholder(tf.int32, [None], "context_dia_tgt_length_%d" % i),
                "context_source": tf.placeholder(tf.int32, [None, None], "context_source_%d" % i),
                "context_source_length": tf.placeholder(tf.int32, [None], "context_source_length_%d" % i),
                "position_dia_src": tf.placeholder(tf.int32, [None, None], "position_dia_src_%d" % i),
                "position_dia_tgt": tf.placeholder(tf.int32, [None, None], "position_dia_tgt_%d" % i),
                "position_ctx_src": tf.placeholder(tf.int32, [None, None], "position_ctx_src_%d" % i),
                "src_image": tf.placeholder(tf.int32, [None, None], "src_image_%d" % i),
                "ctx_src_image": tf.placeholder(tf.int32, [None, None], "ctx_src_image_%d" % i),
                "src_ctx_image": tf.placeholder(tf.int32, [None, None], "src_ctx_image_%d" % i),
                "tgt_ctx_image": tf.placeholder(tf.int32, [None, None], "tgt_ctx_image_%d" % i)
            })

        # A list of outputs
        if params.generate_samples:
            inference_fn = sampling.create_sampling_graph
        else:
            inference_fn = inference.create_inference_graph

        predictions = parallel.data_parallelism(
            params.device_list, lambda f: inference_fn(model_list, f, params),
            placeholders)

        # Create assign ops
        assign_ops = []
        feed_dict = {}

        all_var_list = tf.trainable_variables()

        for i in range(len(args.checkpoints)):
            un_init_var_list = []
            name = model_cls_list[i].get_name()

            for v in all_var_list:
                if v.name.startswith(name + "_%d" % i):
                    un_init_var_list.append(v)

            ops = set_variables(un_init_var_list, model_var_lists[i],
                                name + "_%d" % i, feed_dict)
            assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)
        init_op = tf.tables_initializer()
        results = []

        tf.get_default_graph().finalize()

        # Create session
        with tf.Session(config=session_config(params)) as sess:
            # Restore variables
            sess.run(assign_op, feed_dict=feed_dict)
            sess.run(init_op)

            while True:
                try:
                    feats = sess.run(features)
                    op, feed_dict = shard_features(feats, placeholders,
                                                   predictions)
                    res = sess.run(predictions, feed_dict=feed_dict)
                    results.append(res)
#                    code.interact(local=locals())#print("res:", res)
#                    results.append(sess.run(op, feed_dict=feed_dict))
                    message = "Finished batch %d" % len(results)
                    tf.logging.log(tf.logging.INFO, message)
                except tf.errors.OutOfRangeError:
                    break

        # Convert to plain text
        vocab = params.vocabulary["target"]
        outputs = []
        scores = []

        for result in results:
            for shard in result:
                for item in shard[0]:
                    outputs.append(item.tolist())
                for item in shard[1]:
                    scores.append(item.tolist())

        restored_inputs = []
        restored_outputs = []
        restored_scores = []

        for index in range(len(sorted_inputs)):
            restored_inputs.append(sorted_inputs[sorted_keys[index]])
            restored_outputs.append(outputs[sorted_keys[index]])
            restored_scores.append(scores[sorted_keys[index]])

        # Write to file
        if sys.version_info.major == 2:
            outfile = open(args.output, "w")
        elif sys.version_info.major == 3:
            outfile = open(args.output, "w", encoding="utf-8")
        else:
            raise ValueError("Unkown python running environment!")

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
                else:
                    pattern = "%d ||| %s ||| %s ||| %f\n"
                    source = restored_inputs[count]
                    values = (count, source, decoded, score)
                    outfile.write(pattern % values)

            count += 1
        outfile.close()

if __name__ == "__main__":
    main(parse_args())
