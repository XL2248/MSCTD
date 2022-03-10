# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math,code
import operator

import numpy as np
import tensorflow as tf
import thumt.utils.distribute as distribute


def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):
    """ Batch examples

    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        # Compute boundaries
        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        # Whether the batch size is constant
        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        # The queue to bucket on will be chosen based on maximum length
        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            keep_input=(max_example_length <= max_length)
        )

    return outputs


def get_training_input(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filenames, target_filenames]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])
        img_mask_dataset = tf.data.Dataset.from_tensor_slices(np.memmap(filenames[4], mode="r", dtype=np.bool, shape=(params.image_num, params.object_num)))
        img_dataset = tf.data.Dataset.from_tensor_slices(np.memmap(filenames[3], mode="r", dtype=np.float32, shape=(params.image_num, params.object_num, 2048)))

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, img_dataset, img_mask_dataset))

        if distribute.is_distributed_training_mode():
            dataset = dataset.shard(distribute.size(), distribute.rank())

        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, tgt, img, mask: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values,
                img,
                mask
                #tf.string_split([img]).values
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt, img, mask: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0),
                img,
                mask
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt, img, mask: {
                "source": src,
                "target": tgt,
                "image": img,
                "image_mask": mask,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt)
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )
        index_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["index"]),
            default_value=0
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])
        #features["image"] = index_table.lookup(features["image"])

        # Batching
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["image"] = tf.to_float(features["image"])
        features["image_mask"] = tf.to_float(features["image_mask"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        return features


def sort_input_file(filename, f1, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    with tf.gfile.Open(f1) as fd:
        f1_inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []
    valid_index = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i
        valid_index.append(f1_inputs[index])

    return sorted_keys, sorted_inputs, valid_index


def sort_and_zip_files(names):
    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])

    return [list(x) for x in zip(*sorted_inputs)]

def get_evaluation_input_img(inputs, params):
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []
        print(len(inputs[0]))
        img_mask_dataset = np.memmap(params.dev_object_mask, mode="r", dtype=np.bool, shape=(len(inputs[0]), params.object_num))
        img_dataset = np.memmap(params.dev_object, mode="r", dtype=np.float32, shape=(len(inputs[0]), params.object_num, 2048))
        new_inputs, img, img_mask = [], [], []
        new_inputs.append(inputs[0])
        new_inputs.append(inputs[1])
        new_inputs.append(np.array(img_dataset))
        new_inputs.append(np.array(img_mask_dataset))
        new_inputs.append(inputs[2])
#        print(len(new_inputs[0]), len(new_inputs[3]), new_inputs[3].shape, new_inputs[2].shape)
#        code.interact(local=locals())
        for i, data in enumerate(new_inputs):
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
        #    dataset = dataset.map(lambda x: tf.string_split([x]).values,
        #                          num_parallel_calls=params.num_threads)
            # Append <eos>
            if i == 0 or i == 4:
                print(i)
                dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                      num_parallel_calls=params.num_threads)
                dataset = dataset.map(
                    lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                    num_parallel_calls=params.num_threads
                )
            else:
                dataset = dataset.map(
                    lambda x: x,
                    num_parallel_calls=params.num_threads
                )

            datasets.append(dataset)
#        print(len(inputs))
        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "image_id": x[1],
                "image": x[2],
                "image_mask": x[3],
                "references": x[4:]
            },
            num_parallel_calls=params.num_threads
        )
#        code.interact(local=locals())
        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [tf.Dimension(None)],
                "source_length": [],
                "image_id": [],
                "image": [None, None],
                "image_mask": [None],
                "references": (tf.Dimension(None),) * (len(new_inputs) - 4)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "image_id": params.pad,
                "image": 0.,
                "image_mask": False,
                "references": (params.pad,) * (len(new_inputs) - 4)
            }
        )
#        print(list(dataset.as_numpy_iterator()))
#        code.interact(local=locals())
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )
        index_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["index"]),
            default_value=1
        )
#        code.interact(local=locals())
        features["source"] = src_table.lookup(features["source"])
        features["image"] = tf.to_float(features["image"])
        features["image_mask"] = tf.to_float(features["image_mask"])
#        features["image"] = tf.to_float(tf.convert_to_tensor(np.array(img))) #tf.to_float(features["image"])
#        features["image_mask"] = tf.to_float(tf.convert_to_tensor(np.array(img_mask))) #tf.to_float(features["image_mask"])

    return features

def get_evaluation_input(inputs, params):
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []

        for data in inputs:
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            # Append <eos>
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads
            )
            datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "references": x[1:]
            },
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [tf.Dimension(None)],
                "source_length": [],
                "references": (tf.Dimension(None),) * (len(inputs) - 1)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "references": (params.pad,) * (len(inputs) - 1)
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )

        features["source"] = src_table.lookup(features["source"])

    return features


def get_inference_input(inputs, typ, params):
    if params.generate_samples:
        batch_size = params.sample_batch_size
    else:
        batch_size = params.decode_batch_size

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(inputs)
        )

        if typ == "x":
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)

            # Append <eos>
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads
            )

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda x: {"source": x, "source_length": tf.shape(x)[0]},
            num_parallel_calls=params.num_threads
        )
        if typ == "x":
            dataset = dataset.padded_batch(
                batch_size * len(params.device_list),
                {"source": [tf.Dimension(None)], "source_length": []},
                {"source": params.pad, "source_length": 0}
            )
        if typ == "img":
            dataset = dataset.padded_batch(
                batch_size * len(params.device_list),
                {"source": [None, None], "source_length": []},
                {"source": 0., "source_length": 0}
            )
        if typ == "img_mask":
            dataset = dataset.padded_batch(
                batch_size * len(params.device_list),
                {"source": [None], "source_length": []},
                {"source": False, "source_length": 0}
            )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        if typ == "x":
            features["source"] = src_table.lookup(features["source"])
        else:
            features["source"] = tf.to_float(features["source"])

        return features

def get_relevance_input(inputs, outputs, params):
    # inputs
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(inputs)
    )

    # Split string
    dataset = dataset.map(lambda x: tf.string_split([x]).values,
                          num_parallel_calls=params.num_threads)

    # Append <eos>
    dataset = dataset.map(
        lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
        num_parallel_calls=params.num_threads
    )

    # Convert tuple to dictionary
    dataset = dataset.map(
        lambda x: {"source": x, "source_length": tf.shape(x)[0]},
        num_parallel_calls=params.num_threads
    )

    dataset = dataset.padded_batch(
        params.decode_batch_size,
        {"source": [tf.Dimension(None)], "source_length": []},
        {"source": params.pad, "source_length": 0}
    )

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["source"]),
        default_value=params.mapping["source"][params.unk]
    )
    features["source"] = src_table.lookup(features["source"])

    # outputs
    dataset_o = tf.data.Dataset.from_tensor_slices(
        tf.constant(outputs)
    )

    # Split string
    dataset_o = dataset_o.map(lambda x: tf.string_split([x]).values,
                          num_parallel_calls=params.num_threads)

    # Append <eos>
    dataset_o = dataset_o.map(
        lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
        num_parallel_calls=params.num_threads
    )

    # Convert tuple to dictionary
    dataset_o = dataset_o.map(
        lambda x: {"target": x, "target_length": tf.shape(x)[0]},
        num_parallel_calls=params.num_threads
    )

    dataset_o = dataset_o.padded_batch(
        params.decode_batch_size,
        {"target": [tf.Dimension(None)], "target_length": []},
        {"target": params.pad, "target_length": 0}
    )

    iterator = dataset_o.make_one_shot_iterator()
    features_o = iterator.get_next()

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["target"]),
        default_value=params.mapping["target"][params.unk]
    )
    features["target"] = src_table.lookup(features_o["target"])
    features["target_length"] = features_o["target_length"]

    return features