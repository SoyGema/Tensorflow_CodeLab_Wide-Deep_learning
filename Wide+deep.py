# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# This code is for DEVFEST GDGSpain using character prediction from Tensorflow
# https://github.com/bigpress/gameofthrones/blob/master/character-predictions.csv
#

# ==============================================================================

"""Import Python 2-3 compatibility glue, ETL (pandas) and ML (TensorFlow/sklearn) libraries"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#Import sklearn metrics for the accuracy 
from sklearn.metrics import accuracy_score
from sklearn import cross_validation # to split the train/test cases

import tempfile
from six.moves import urllib

import numpy as np
import pandas as pd
import tensorflow as tf


## Begin set up logging
import logging

logger = logging.getLogger('net_mk1')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
## End set up logging


# Get some useful information from TensorFlow's internals
# tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.INFO)


"""Flags are a TensorFlow internal util to define command-line parameters."""
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
# flags.DEFINE_integer("snapshot_steps", 10, "Number of steps between snapshots.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

flags.DEFINE_float("wide_learning_rate", 0.001, "learning rate for the wide part of the model")
flags.DEFINE_float("deep_learning_rate", 0.003, "learning rate for the deep part of the model")

learning_rate = [FLAGS.wide_learning_rate, FLAGS.deep_learning_rate]
model_name = "net_mk1"



# Target column is plod, for Percentage Likelyhood Of Death
LABEL_COLUMN = 'plod'

# The columns in the dataset are the following:
COLUMNS = 'S.No,actual,pred,alive,plod,name,title,male,culture,dateOfBirth,mother,father,heir,house,spouse,book1,book2,book3,book4,book5,isAliveMother,isAliveFather,isAliveHeir,isAliveSpouse,isMarried,isNoble,age,numDeadRelations,boolDeadRelations,isPopular,popularity,isAlive'.split(',')
COLUMNS_X = [col for col in COLUMNS if col != LABEL_COLUMN]

dataset_file_name = "../dataset/character-predictions.csv"

df_base = pd.read_csv(dataset_file_name, sep=',',) # names=COLUMNS, skipinitialspace=True, skiprows=1)

CATEGORICAL_COLUMN_NAMES = [
    'male',
    'culture',
    'mother',
    'father',
    'title',
    'heir',
    'house',
    'spouse',
    'numDeadRelations',
    'boolDeadRelations',
]

BINARY_COLUMNS = [
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
    'isPopular',
]

for col in BINARY_COLUMNS:
  df_base[col] = df_base[col].astype(str)

BUCKETIZED_COLUMNS_FILTER = ['age']

CATEGORICAL_COLUMNS = {
    col: len(df_base[col].unique()) + 1
    for col in CATEGORICAL_COLUMN_NAMES
}

# TODO: Revise bins for GoT
AGE_BINS = [ 0, 4, 8, 15, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 80, 65535 ]

CONTINUOUS_COLUMNS = [
  'age',
  'popularity',
  'dateOfBirth',
]

UNUSED_COLUMNS = [
  col
  for col in COLUMNS
  if col not in CONTINUOUS_COLUMNS and col not in BINARY_COLUMNS and col not in CATEGORICAL_COLUMN_NAMES
]

X = df_base[COLUMNS]
y = df_base[LABEL_COLUMN]

df_train, df_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)

LABEL_COLUMN_CLASSES = y.unique()

dataset_file_name = "../dataset/character-predictions.csv"


def build_estimator(model_dir):
  """Build an estimator."""

  logger.info("Learning rates (wide, deep) = %s", learning_rate)

  wide_columns = get_wide_columns()
  deep_columns = get_deep_columns()

  ##From here, reading the code beyond i've change anything from the tutorial 
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
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m


age_column = tf.contrib.layers.real_valued_column('age', dimension=1, dtype=tf.int32)

def get_deep_columns():
  cc_input_var = {}
  cc_embed_var = {}
  cols = [] + [age_column]

  for cc in BINARY_COLUMNS:
    cols.append(
      tf.contrib.layers.embedding_column(
        tf.contrib.layers.sparse_column_with_keys(
          column_name=cc,
          keys=["0", "1"],
        ),
        dimension=8)
    )

  for cc, cc_size in CATEGORICAL_COLUMNS.items():
    cc_input_var[cc] = tf.contrib.layers.embedding_column(
      tf.contrib.layers.sparse_column_with_hash_bucket(
        cc,
        hash_bucket_size=cc_size,
      ),
      dimension=8
    )

    cols.append(cc_input_var[cc])
    # cols.append(tf.squeeze(cc_input_var[cc], squeeze_dims=[1]))

    # with tf.scope_name('%s_in' % cc):
    #   cc_input_var[cc] = tf.placeholder(shape=[None, 1], dtype=tf.int32)
    # cc_embed_var[cc] = tflearn.layers.embedding_ops.embedding(cc_input_var[cc], cc_size, 10, name='deep_%s_embed' % cc)
    # logger.debug("%s_embed = %s", cc, cc_embed_var)
    # 
    # cols.append(tf.squeeze(cc_embed_var[cc], squeeze_dims=[1], name="%s_squeeze" % cc))

  # age_buckets = tf.contrib.layers.bucketized_column(age,
  #                                                   boundaries=[
  #                                                       18, 25, 30, 35, 40, 45,
  #                                                       50, 55, 60, 65
  #                                                   ])
  ##Crossed clomuns come in pairs or can I combined more??

  return cols


def get_wide_columns():
  cols = []
  for column in CONTINUOUS_COLUMNS:
    cols.append(tf.contrib.layers.real_valued_column(column, dimension=1, dtype=tf.float32))

  # cols.append(age_column)
  # cols.append(tf.contrib.layers.bucketized_column(age_column, boundaries=AGE_BINS))

  logger.info("Got wide columns %s", cols)
  return cols


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  def mp(tensor):
    """Monkey-patch tensor to include get_shape"""
    from tensorflow.python.framework import tensor_util
    shape = tensor_util.constant_value_as_shape(tensor._shape)
    tensor.get_shape = lambda: shape
    return tensor

  categorical_cols = {
    k: mp(tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],
                       values=df[k].values,
                       shape=[df[k].size, 1]))
    for k in (list(CATEGORICAL_COLUMNS.keys()) + BINARY_COLUMNS)
  }
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval():
  """Train and evaluate the model."""
  
  df_base = pd.read_csv(
    tf.gfile.Open(dataset_file_name),
    names=COLUMNS,
    skipinitialspace=True,
    skiprows=1,
    engine="python")

  ## Remove NaN elements
  # df_base = df_base.dropna(how='any', axis=0)

  ## Fill NaN elements
  for col in CATEGORICAL_COLUMN_NAMES:
    df_base[col] = np.where(df_base[col].isnull(), 'NULL', df_base[col])
  for col in BINARY_COLUMNS:
    df_base[col] = np.where(df_base[col].isnull(), "0", df_base[col])
  for col in CONTINUOUS_COLUMNS:
    df_base[col] = np.where(df_base[col].isnull(), 0., df_base[col])

  for col in UNUSED_COLUMNS:
    df_base[col] = np.where(df_base[col].isnull(), 0, df_base[col])

  logger.debug("Number of columns after removing nulls: %d (before: %d)", len(df_base.dropna(how='any', axis=0)), len(df_base))

  df_base[LABEL_COLUMN] = (
      df_base["isAlive"].apply(lambda x: x)).astype(int)

  df_train, df_test = cross_validation.train_test_split(df_base, test_size=0.2, random_state=42)
  # df_train = pd.read_csv(
  #     tf.gfile.Open(train_file_name),
  #     names=COLUMNS,
  #     skipinitialspace=True,
  #     engine="python")
  # df_test = pd.read_csv(
  #     tf.gfile.Open(test_file_name),
  #     names=COLUMNS,
  #     skipinitialspace=True,
  #     skiprows=1,
  #     engine="python")

  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir)
  m.fit(
    input_fn=lambda: input_fn(df_train),
    # snapshot_step=FLAGS.snapshot_steps,
    steps=FLAGS.train_steps
  )
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
