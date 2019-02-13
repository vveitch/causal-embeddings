"""

Parses PeerRead data into a Bert-based model compatible format, and stores as tfrecord

See dataset.py for the corresponding code to read this data

"""
import glob
import numpy as np
import tensorflow as tf
import argparse
try:
    import mkl_random as random
except ImportError:
    import numpy.random as random

import bert.tokenization as tokenization
from dataset.sentence_masking import create_masked_lm_predictions

# hardcoded because protobuff is not self describing for some bizarre reason
all_context_features = \
    {'accepted': tf.int64,
     'most_recent_reference_year': tf.int64,
     'num_recent_references': tf.int64,
     'num_references': tf.int64,
     'num_refmentions': tf.int64,
     # 'avg_length_reference_mention_contexts': tf.float32,
     'abstract_contains_deep': tf.int64,
     'abstract_contains_neural': tf.int64,
     'abstract_contains_embedding': tf.int64,
     'abstract_contains_outperform': tf.int64,
     'abstract_contains_novel': tf.int64,
     'abstract_contains_state-of-the-art': tf.int64,
     "title_contains_deep": tf.int64,
     "title_contains_neural": tf.int64,
     "title_contains_embedding": tf.int64,
     "title_contains_gan": tf.int64,
     'num_ref_to_figures': tf.int64,
     'num_ref_to_tables': tf.int64,
     'num_ref_to_sections': tf.int64,
     'num_uniq_words': tf.int64,
     'num_sections': tf.int64,
     # 'avg_sentence_length': tf.float32,
     'contains_appendix': tf.int64,
     'title_length': tf.int64,
     'num_authors': tf.int64,
     'num_ref_to_equations': tf.int64,
     'num_ref_to_theorems': tf.int64,
     'id': tf.int64,
     'year': tf.int64,
     'venue': tf.int64,
     'many_split': tf.int64}


def compose(*fns):
    """ Composes the given functions in reverse order.

    Parameters
    ----------
    fns: the functions to compose

    Returns
    -------
    comp: a function that represents the composition of the given functions.
    """
    import functools

    def _apply(x, f):
        if isinstance(x, tuple):
            return f(*x)
        else:
            return f(x)

    def comp(*args):
        return functools.reduce(_apply, fns, args)

    return comp


def make_parser(abs_seq_len=250):
    context_features = {k: tf.FixedLenFeature([], dtype=v) for k, v in all_context_features.items()}

    abstract_features = {
        "token_ids": tf.FixedLenFeature([abs_seq_len], tf.int64),
        "token_mask": tf.FixedLenFeature([abs_seq_len], tf.int64),
        # "segment_ids": tf.FixedLenFeature([abs_seq_len], tf.int64),
    }

    _name_to_features = {**context_features, **abstract_features}

    def parser(record):
        tf_example = tf.parse_single_example(
            record,
            features=_name_to_features
        )

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(tf_example.keys()):
            t = tf_example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            tf_example[name] = t


        return tf_example

    return parser


def make_input_id_masker(tokenizer, seed):
    # (One of) Bert's unsupervised objectives is to mask some fraction of the input words and predict the masked words

    def masker(data):
        token_ids = data['token_ids']
        maybe_masked_token_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = create_masked_lm_predictions(
            token_ids,
            # pre-training defaults from Bert docs
            masked_lm_prob=0.15,
            max_predictions_per_seq=20,
            vocab=tokenizer.vocab,
            seed=seed)
        return {
            **data,
            'maybe_masked_token_ids': maybe_masked_token_ids,
            'masked_lm_positions': masked_lm_positions,
            'masked_lm_ids': masked_lm_ids,
            'masked_lm_weights': masked_lm_weights
        }

    return masker


def make_extra_feature_cleaning():
    def extra_feature_cleaning(data):
        data['num_authors'] = tf.minimum(data['num_authors'], 6)-1
        data['year'] = data['year']-2007

        # some extras
        equation_referenced = tf.minimum(data['num_ref_to_equations'], 1)
        theorem_referenced = tf.minimum(data['num_ref_to_theorems'], 1)

        # buzzy title
        any_buzz = data["title_contains_deep"] + data["title_contains_neural"] + \
                   data["title_contains_embedding"] + data["title_contains_gan"]
        buzzy_title = tf.cast(tf.not_equal(any_buzz, 0), tf.int32)

        return {**data,
                'equation_referenced': equation_referenced,
                'theorem_referenced': theorem_referenced,
                'buzzy_title': buzzy_title}
    return extra_feature_cleaning


def make_label():
    """
    Do something slightly nuts for testing purposes
    :return:
    """
    def labeler(data):
        return {**data, 'label_ids': data['accepted']}

    # def wacky_labeler(data):
    #     label_ids = tf.greater_equal(data['num_authors'], 4)
    #     label_ids = tf.cast(label_ids, tf.int32)
    #     return {**data, 'label_ids': label_ids}

    return labeler


def make_split_document_labels(num_splits, dev_splits, test_splits):
    """
    Adapts tensorflow dataset to produce additional elements that indicate whether each datapoint is in train, dev,
    or test

    Particularly, splits the data into num_split folds, and censors the censored_split fold

    Parameters
    ----------
    num_splits integer in [0,100)
    dev_splits list of integers in [0,num_splits)
    test_splits list of integers in [0, num_splits)

    Returns
    -------
    fn: A function that can be used to map a dataset to censor some of the document labels.
    """
    def _tf_in1d(a,b):
        """
        Tensorflow equivalent of np.in1d(a,b)
        """
        a = tf.expand_dims(a, 0)
        b = tf.expand_dims(b, 1)
        return tf.reduce_any(tf.equal(a, b), 1)

    def _tf_scalar_a_in1d_b(a, b):
        """
        Tensorflow equivalent of np.in1d(a,b)
        """
        return tf.reduce_any(tf.equal(a, b))

    def fn(data):
        many_split = data['many_split']
        reduced_split = tf.floormod(many_split, num_splits)  # reduce the many splits to just num_splits

        in_dev = _tf_scalar_a_in1d_b(reduced_split, dev_splits)
        in_test = _tf_scalar_a_in1d_b(reduced_split, test_splits)
        in_train = tf.logical_not(tf.logical_or(in_dev, in_test))

        # in_dev = _tf_in1d(reduced_splits, dev_splits)
        # in_test = _tf_in1d(reduced_splits, test_splits)
        # in_train = tf.logical_not(tf.logical_or(in_dev, in_test))

        # code expects floats
        in_dev = tf.cast(in_dev, tf.float32)
        in_test = tf.cast(in_test, tf.float32)
        in_train = tf.cast(in_train, tf.float32)

        return {**data, 'in_dev': in_dev, 'in_test': in_test, 'in_train': in_train}

    return fn


def dataset_processing(dataset, parser, masker, labeler, is_training, num_splits, dev_splits, test_splits, batch_size,
                       filter_test=False,
                       shuffle_buffer_size=100):
    """

    Parameters
    ----------
    dataset  tf.data dataset
    parser function, read the examples, should be based on tf.parse_single_example
    masker function, should provide Bert style masking
    labeler function, produces labels
    is_training
    num_splits
    censored_split
    batch_size
    filter_test restricts to only examples where in_test=1
    shuffle_buffer_size

    Returns
    -------

    """

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    data_processing = compose(parser,  # parse from tf_record
                              labeler,  # add a label (unused downstream at time of comment)
                              make_split_document_labels(num_splits, dev_splits, test_splits),  # censor some labels
                              masker)  # Bert style token masking for unsupervised training

    dataset = dataset.map(data_processing, 4)

    if filter_test:
        def filter_test_fn(data):
            return tf.equal(data['in_test'], 1)

        dataset = dataset.filter(filter_test_fn)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    return dataset


def make_input_fn_from_file(input_files_or_glob, seq_length,
                            num_splits, dev_splits, test_splits,
                            tokenizer, is_training,
                            filter_test=False,
                            shuffle_buffer_size=100, seed=0):

    input_files = []
    for input_pattern in input_files_or_glob.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))


    def input_fn(params):
        batch_size = params["batch_size"]

        if is_training:
            dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=len(input_files))
            cycle_length = min(4, len(input_files))

        else:
            dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            cycle_length = 1  # go through the datasets in a deterministic order

        # make the record parsing ops
        max_abstract_len = seq_length

        parser = make_parser(max_abstract_len)  # parse the tf_record
        parser = compose(parser, make_extra_feature_cleaning())
        masker = make_input_id_masker(tokenizer, seed)  # produce masked subsets for unsupervised training
        labeler = make_label()

        # for use with interleave
        def _dataset_processing(input):
            input_dataset = tf.data.TFRecordDataset(input)
            processed_dataset = dataset_processing(input_dataset,
                                                   parser, masker, labeler,
                                                   is_training,
                                                   num_splits, dev_splits, test_splits,
                                                   batch_size, filter_test, shuffle_buffer_size)
            return processed_dataset



        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                _dataset_processing,
                sloppy=is_training,
                cycle_length=cycle_length))

        return dataset

    return input_fn


def main():
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_buffer_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_abs_len', type=int, default=250)

    args = parser.parse_args()

    # for easy debugging
    # filename = "../../data/PeerRead/proc/acl_2017.tf_record"
    # filename = glob.glob('../../data/PeerRead/proc/*.tf_record')
    filename = '../../data/PeerRead/proc/arxiv*.tf_record'

    vocab_file = "../../BERT_pre-trained/uncased_L-12_H-768_A-12/vocab.txt"
    seed = 0
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    num_splits = 10
    dev_splits = []
    test_splits = [1, 2]

    input_dataset_from_filenames = make_input_fn_from_file(filename,
                                                           250,
                                                           num_splits,
                                                           dev_splits,
                                                           test_splits,
                                                           tokenizer,
                                                           is_training=False,
                                                           filter_test=True,
                                                           seed=0)
    params = {'batch_size': 16}
    input_dataset = input_dataset_from_filenames(params)

    # masker = make_input_id_masker(tokenizer, seed)
    # parser = make_parser(args.max_abs_len)
    # labeler = make_label()
    #
    # dataset = tf.data.TFRecordDataset(filename)

    # input_dataset = dataset_processing(dataset, parser, masker, labeler,
    #                                    is_training=True, num_examples=num_examples, split_indices=split_indices,
    #                                    batch_size=args.batch_size, shuffle_buffer_size=100)

    secs = []

    itr = input_dataset.make_one_shot_iterator()
    # print(itr.get_next()["token_ids"].name)
    for i in range(25):
        sample = itr.get_next()
        # "title_contains_deep": tf.int64,
        # "title_contains_neural": tf.int64,
        # "title_contains_embedding": tf.int64,
        # "title_contains_gan": tf.int64,
        print(sample['buzzy_title'])
        # print(sample['venue'])
        # print(sample['in_test'])
        # print(i)
        # secs += [sample['venue']]
        # print(sample['in_dev'])
        # print(sample['masked_lm_weights'])
        # print(tf.reduce_sum(sample['masked_lm_weights'],1))
        # print(tf.reduce_mean(tf.reduce_sum(sample['masked_lm_weights'],1)))

        # print(sample)
        # print(np.max([np.max(secs_batch) for secs_batch in secs]))
        # print(np.unique(np.concatenate(secs), return_counts=True))

if __name__ == "__main__":
    main()