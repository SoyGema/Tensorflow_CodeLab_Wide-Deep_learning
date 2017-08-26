#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2016 The TF Codelab Contributors. All Rights Reserved.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#   http://www.apache.org/licenses/LICENSE-2.0
# This code was originally presented at GDGSpain DevFest
# using character prediction from Tensorflow
# https://github.com/bigpress/gameofthrones/blob/master/character-predictions.csv
#
# Latest version is always available at:
# https://github.com/codelab-tf-got/code/
# Codelab test is available at: https://codelab-tf-cot.github.io
# Codelab code  by @ssice . Front @SoyGema
# ==============================================================================
"""TensorFlow codelab GoT and Wide+Deep."""

# Import Python 2-3 compatibility glue, ETL (pandas) and ML
# (TensorFlow/sklearn) libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils,
    input_fn_utils
)
from sklearn import cross_validation  # to split the train/test cases

logger = logging.getLogger('net_mk1')
# Uncomment the logging lines to see logs in the console
# to get to know better what this code does
'''
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -\
                              %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
'''
# End set up logging

# Stop tensorflow from getting chatty with us
tf.logging.set_verbosity(tf.logging.ERROR)
# tf.logging.set_verbosity(tf.logging.WARN)
# tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None
model_name = "net_mk2"


def only_existing(l, haystack):
    """Helper to filter elements not already on the haystack in O(n)."""
    s = set(haystack)
    return [item for item in l if item in s]


def get_dataset(filename, local_path='../dataset'):
    """Download a dataset for this codelab.

    The buckets here are managed by @ssice.
    """
    gcs_base = 'https://storage.googleapis.com/'
    gcs_path = 'codelab-got.appspot.com/dataset/'
    return base.maybe_download(filename,
                               local_path,
                               gcs_base + gcs_path + filename)


##############################################################################
# Column definitions
##############################################################################

# The columns in the dataset are the following:
COLUMNS = ['S.No',
           'actual',
           'pred',
           'alive',
           'plod',
           'name',
           'title',
           'male',
           'culture',
           'dateOfBirth',
           'mother',
           'father',
           'heir',
           'house',
           'spouse',
           'book1',
           'book2',
           'book3',
           'book4',
           'book5',
           'isAliveMother',
           'isAliveFather',
           'isAliveHeir',
           'isAliveSpouse',
           'isMarried',
           'isNoble',
           'age',
           'numDeadRelations',
           'boolDeadRelations',
           'isPopular',
           'popularity',
           'isAlive']

dataset_file_name = get_dataset('character-predictions.csv', '../dataset')


# :: UNCOMMENT for use in the alternative dataset ::
"""
COLUMNS = ['S.No',
           'name',
           'title',
           'male',
           'culture',
           'house',
           'book1',
           'book2',
           'book3',
           'book4',
           'book5',
           'isAliveMother',
           'isAliveFather',
           'isAliveHeir',
           'isAliveSpouse',
           'isMarried',
           'isNoble',
           'numDeadRelations',
           'boolDeadRelations',
           'popularity',
           'isAlive']
dataset_file_name = get_dataset('character-curated.csv', '../dataset')
"""

# Target column is the actual isAlive variable
LABEL_COLUMN = 'isAlive'

COLUMNS_X = [col for col in COLUMNS if col != LABEL_COLUMN]

CATEGORICAL_COLUMN_NAMES = only_existing(['male',
                                          'culture',
                                          'mother',
                                          'father',
                                          'title',
                                          'heir',
                                          'house',
                                          'spouse',
                                          'numDeadRelations',
                                          'boolDeadRelations'],
                                         COLUMNS)

BINARY_COLUMNS = only_existing(['book1',
                                'book2',
                                'book3',
                                'book4',
                                'book5',
                                'isAliveMother',
                                'isAliveFather',
                                'isAliveHeir',
                                'isAliveSpouse',
                                'isMarried',
                                'isNoble',
                                'isPopular'],
                               COLUMNS)

CONTINUOUS_COLUMNS = only_existing(['age',
                                    'popularity',
                                    'dateOfBirth'],
                                   COLUMNS)

FEATURE_COLUMNS = [
    col for col in COLUMNS
    if col in CONTINUOUS_COLUMNS
    or col in BINARY_COLUMNS
    or col in CATEGORICAL_COLUMN_NAMES
]

UNUSED_COLUMNS = [
    col
    for col in COLUMNS
    if col != LABEL_COLUMN and col not in FEATURE_COLUMNS
]

print("We are not using columns: %s" % UNUSED_COLUMNS)


# Load the base dataframe
df_base = pd.read_csv(dataset_file_name,
                      sep=',',
                      names=COLUMNS,
                      skipinitialspace=True,
                      skiprows=1)


# We re-type the binary columns so that they are strings
for col in BINARY_COLUMNS:
    df_base[col] = df_base[col].astype(str)


# We get, for each categorical column, the number of unique elements
# it has.
CATEGORICAL_COLUMNS = {
    col: len(df_base[col].unique()) + 1
    for col in CATEGORICAL_COLUMN_NAMES
}

"""
preset_deep_columns = [tf.contrib.layers.real_valued_column('age',
                                                            dimension=1,
                                                            dtype=tf.int32)]
"""
preset_deep_columns = []


def get_deep_columns():
    """Obtain the deep columns of the model.

    In our model, these are the binary columns (which are embedded with
    keys "0" and "1") and the categorical columns, which are embedded as
    8-dimensional sparse columns in hash buckets.
    """
    cc_input_var = {}
    cols = preset_deep_columns

    for cc in BINARY_COLUMNS:
        cols.append(
            tf.contrib.layers.embedding_column(
                tf.contrib.layers.sparse_column_with_keys(column_name=cc,
                                                          keys=["0", "1"]),
                dimension=8)
            )

    for cc, cc_size in CATEGORICAL_COLUMNS.items():
        cc_input_var[cc] = tf.contrib.layers.embedding_column(
            tf.contrib.layers.sparse_column_with_hash_bucket(
                cc,
                hash_bucket_size=cc_size
            ),
            dimension=8
        )
        cols.append(cc_input_var[cc])

    for column in CONTINUOUS_COLUMNS:
        cols.append(tf.contrib.layers.real_valued_column(column,
                                                         dimension=1,
                                                         dtype=tf.float32))

    return cols


def get_wide_columns():
    """Get wide columns for our model.

    In this case, wide columns are just the continuous columns.
    """
    cols = []
    for column in CONTINUOUS_COLUMNS:
        cols.append(tf.contrib.layers.real_valued_column(column,
                                                         dimension=1,
                                                         dtype=tf.float32))

    logger.info("Got wide columns %s", cols)
    return cols


##############################################################################
# General estimator builder function
# The wide/deep part construction is below. This gathers both parts
# and joins the model into a single classifier.
##############################################################################
def build_estimator(model_dir):
    """General estimator builder function.

    The wide/deep part construction is below. This gathers both parts
    and joins the model into a single classifier.
    """
    wide_columns = get_wide_columns()
    deep_columns = get_deep_columns()

    if FLAGS.model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif FLAGS.model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=None,  # WATCH: Linear optimizer. By default, FTRL
            dnn_feature_columns=deep_columns,
            dnn_activation_fn=None,  # WATCH: Activation function for DNN
                                     # (default: relu)
            dnn_hidden_units=[100, 50],  # WATCH: Hidden units for the DNN part
            dnn_dropout=None,  # WATCH: Dropout for the DNN
            dnn_optimizer=None,  # WATCH: Optimizer for DNN
                                 # (Adagrad by default)
            fix_global_step_increment_bug=True
            )
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k)
    # to the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column
    # name(k) to the values of that column stored in a tf.SparseTensor.

    """
    Categorical columns go into sparse tensors because there are just
    sparse values here, and using a dense tensor would be a waste of resources
    """
    categorical_cols = {
        k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],
                           values=df[k].values,
                           dense_shape=[df[k].size, 1])
        for k in (list(CATEGORICAL_COLUMNS.keys()) + BINARY_COLUMNS)
    }

    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)

    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)

    # Returns the feature columns and the label.
    return feature_cols, label


def generate_input_fn(df):
    """Generate the input_fn."""
    def _input_fn():
        """Input builder function."""
        # Creates a dictionary mapping from each continuous feature
        # column name (k) to the values of that column stored in a
        # constant Tensor.
        continuous_cols = {k: tf.constant(df[k].values)
                           for k in CONTINUOUS_COLUMNS}

        # Creates a dictionary mapping from each categorical
        # feature column name (k) to the values of that column stored
        # in a tf.SparseTensor.
        categorical_cols = {
            k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],
                               values=df[k].values,
                               dense_shape=[df[k].size, 1])
            for k in (list(CATEGORICAL_COLUMNS.keys()) + BINARY_COLUMNS)
        }

        # Merges the two dictionaries into one.
        feature_cols = dict(continuous_cols)
        feature_cols.update(categorical_cols)

        # Converts the label column into a constant Tensor.
        label = tf.constant(df[LABEL_COLUMN].values)

        # Returns the feature columns and the label.
        return feature_cols, label
    return _input_fn


def column_to_dtype(column):
    """Get the TF data type of the column."""
    if column == LABEL_COLUMN:
        return tf.int32
    elif column in CATEGORICAL_COLUMNS or column in BINARY_COLUMNS:
        return tf.string
    return tf.float32


def serving_input_fn():
    """Serve the input_fn."""
    feature_placeholders = {
        column: tf.placeholder(column_to_dtype(column), [None])
        for column in FEATURE_COLUMNS
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    return input_fn_utils.InputFnOps(
        features,
        None,
        feature_placeholders
    )


def generate_experiment(output_dir, df_train, df_test):
    """Generate the experiment to run."""
    def _experiment_fn(output_dir):
        my_model = build_estimator(output_dir)
        experiment = tf.contrib.learn.Experiment(
            my_model,
            train_input_fn=generate_input_fn(df_train),
            eval_input_fn=generate_input_fn(df_test),
            train_steps=FLAGS.steps,
            export_strategies=[saved_model_export_utils.make_export_strategy(
                serving_input_fn,
                default_output_alternative_key=None
            )]
        )
        return experiment
    return _experiment_fn


def fill_dataframe(dataframe):
    """Fill the missing values with a NaN.

    Fill with a NaN element of the correct type to have a valid label
    to use in the neuron pipeline
    """
    for column in CATEGORICAL_COLUMN_NAMES:
        dataframe[column] = np.where(dataframe[column].isnull(),
                                     'NULL',
                                     dataframe[column])

    for column in BINARY_COLUMNS:
        dataframe[column] = np.where(dataframe[column].isnull(),
                                     "0",
                                     dataframe[column])

    for column in CONTINUOUS_COLUMNS:
        dataframe[column] = np.where(dataframe[column].isnull(),
                                     0.,
                                     dataframe[column])

    for column in UNUSED_COLUMNS:
        dataframe[column] = np.where(dataframe[column].isnull(),
                                     0,
                                     dataframe[column])


def train_and_eval(job_dir=None):
    """Train and evaluate the model."""
    fill_dataframe(df_base)
    logger.debug("Number of columns after removing nulls: %d (before: %d)",
                 len(df_base.dropna(how='any', axis=0)),
                 len(df_base))

    df_base[LABEL_COLUMN] = (
        df_base[LABEL_COLUMN].apply(lambda x: x)).astype(int)

    df_train, df_test = cross_validation.train_test_split(df_base,
                                                          test_size=0.2,
                                                          random_state=42)

    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print("model directory = %s" % model_dir)

    if FLAGS.training_mode == 'manual':
        m = build_estimator(model_dir)
        m.fit(
            input_fn=lambda: input_fn(df_train),
            steps=FLAGS.steps
            )
        results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
        for key in sorted(results):
            print("%s: %s" % (key, results[key]))

    elif FLAGS.training_mode == 'learn_runner':
        experiment_fn = generate_experiment(model_dir,
                                            df_train,
                                            df_test)

        metrics, output_folder = learn_runner.run(experiment_fn, model_dir)
        for key in sorted(metrics):
            print("%s: %s" % (key, metrics[key]))
        print('Model exported to {}'.format(output_folder))


def main(_):
    """Main of the program."""
    train_and_eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_mode",
                        type=str,
                        default="learn_runner",
                        help="Mode to use for training (learn_runner \
                            or manual).")

    parser.add_argument("--model_dir",
                        type=str,
                        default="",
                        help="Base directory for output models.")

    parser.add_argument("--model_type",
                        type=str,
                        default="wide_n_deep",
                        help="Valid model types: {'wide', 'deep', \
                            'wide_n_deep'}.")

    parser.add_argument("--steps",
                        type=int,
                        default=200,
                        help="Number of training steps.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
