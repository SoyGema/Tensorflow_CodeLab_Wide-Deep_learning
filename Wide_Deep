# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# This code is for DEVFEST GDGSpain using character prediction from Tensorflow
# https://github.com/bigpress/gameofthrones/blob/master/character-predictions.csv
#

# ==============================================================================

"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#Import sklearn metrics for the accuracy 
from sklearn.metrics import accuracy_score

import tempfile
from six.moves import urllib

import pandas as pd
import tensorflow as tf

## What is a flag for ? 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

## Columns define all columns of the data set 
COLUMNS = ["S.No", "alive", "name",
           "title", "male", "culture", "dateOfBirth", "DateoFdeath",
           "mother", "father", "heir", "house", "spouse",
           "book1", "book2", "book3", "book4", "book5",
           "isAliveMother", "isAliveFather", "isAliveHeir", "isAliveSpouse",
           "isMarried", "isNoble", "age", "numDeadRelations", "boolDeadRelations"
           "isPopular", "popularity", "isAlive"]
LABEL_COLUMN = "label"
## What is the difference in between categorical and continuous columns ?
## One column can be categorical and continuous how to know the difference? How to choose data for both?
## So this assumption is important right ? -----wide=continuous : deep=categorical -----
## In this case I have put the data as both categorical and continuous : "house","title","numDeadRelations"
CATEGORICAL_COLUMNS = ["alive", "title", "male", "culture",
                       "house", "spouse", "isAliveMother", "isAliveFather", "isAliveHeir",
                       "isAliveSpouse", "isMarried", "isNoble", "numDeadRelations",
                       "boolDeadRelations", "isPopular" , "popularity"]
CONTINUOUS_COLUMNS = ["name", "dateOfBirth", "DateoFdeath", "mother", "father",
                      "heir", "book1", "book2", "book3", "book4", "book5", "age",
                      "isAlive", "house", "title", "numDeadRelations"]

train_file_name = pandas.read_csv( "path")
test_file_name = pandas.read_csv( "path")

##What does an estimator do exactly?
## In the estimator ,we sparse categorical and continuous?

def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.
  alive = tf.contrib.layers.sparse_column_with_hash_bucket(
      "alive", hash_bucket_size=100)
  title = tf.contrib.layers.sparse_column_with_hash_bucket(
      "title", hash_bucket_size=100)
  culture = tf.contrib.layers.sparse_column_with_hash_bucket(
      "culture", hash_bucket_size=100)
  house = tf.contrib.layers.sparse_column_with_hash_bucket(
      "house", hash_bucket_size=100)
  spouse = tf.contrib.layers.sparse_column_with_hash_bucket(
      "spouse", hash_bucket_size=100)
  male = tf.contrib.layers.sparse_column_with_keys(column_name="male",
                                                     keys=["1", "0"])
  isAliveMother = tf.contrib.layers.sparse_column_with_keys(column_name="isAliveMother",
                                                     keys=["1", "0"])
  isAliveFather = tf.contrib.layers.sparse_column_with_keys(column_name="isAliveFather",
                                                     keys=["1", "0"])
  isAliveHeir = tf.contrib.layers.sparse_column_with_keys(column_name="isAliveHeir",
                                                     keys=["1", "0"])  
  isAliveSpouse = tf.contrib.layers.sparse_column_with_keys(column_name="isAliveSpouse",
                                                     keys=["1", "0"])
  isMarried = tf.contrib.layers.sparse_column_with_keys(column_name="isMarried",
                                                     keys=["1", "0"])
  isNoble = tf.contrib.layers.sparse_column_with_keys(column_name="isNoble",
                                                     keys=["1", "0"])
  isPopular = tf.contrib.layers.sparse_column_with_keys(column_name="isPopular",
                                                     keys=["1", "0"])

  numDeadRelations = tf.contrib.layers.sparse_column_with_hash_bucket(
      "numDeadRelations", hash_bucket_size=100)
  boolDeadRelations = tf.contrib.layers.sparse_column_with_hash_bucket(
      "boolDeadRelations", hash_bucket_size=100)
  isPopular = tf.contrib.layers.sparse_column_with_hash_bucket(
      "isPopular", hash_bucket_size=100)
  popularity = tf.contrib.layers.sparse_column_with_hash_bucket(
      "popularity", hash_bucket_size=100)

  # Continuous base columns.
  name = tf.contrib.layers.real_valued_column("name")
  dateOfBirth = tf.contrib.layers.real_valued_column("dateOfBirth")
  DateoFdeath = tf.contrib.layers.real_valued_column("DateoFdeath")
  mother = tf.contrib.layers.real_valued_column("mother")
  father = tf.contrib.layers.real_valued_column("father")
  heir = tf.contrib.layers.real_valued_column("heir")
  book1 = tf.contrib.layers.real_valued_column("book1")
  book2 = tf.contrib.layers.real_valued_column("book2")  
  book3 = tf.contrib.layers.real_valued_column("book3")
  book4 = tf.contrib.layers.real_valued_column("book4")
  book5 = tf.contrib.layers.real_valued_column("book5")
  age = tf.contrib.layers.real_valued_column("age")
  isAlive = tf.contrib.layers.real_valued_column("isAlive")

  house = tf.contrib.layers.real_valued_column("house")
  title = tf.contrib.layers.real_valued_column("title")
  numDeadRelations = tf.contrib.layers.real_valued_column("numDeadRelations")
  
  ## I basically don´t understand this age_bucket transformation, either what it does or what´s for 
  # Transformations
  age_buckets = tf.contrib.layers.bucketized_column(age,
                                                    boundaries=[
                                                        18, 25, 30, 35, 40, 45,
                                                        50, 55, 60, 65
                                                    ])
  ##Crossed clomuns come in pairs or can I combined more??
  # Wide columns and deep columns.
  wide_columns = [name, dateOfBirth, DateoFdeath, mother, father, heir, book1, book2,
                  book3, book4, book5, age, isAlive
                  tf.contrib.layers.crossed_column([house, title],
                                                   hash_bucket_size=int(1e4)),
                  tf.contrib.layers.crossed_column(
                      [age_buckets, house, title],
                      hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([numDeadRelations, title],
                                                   hash_bucket_size=int(1e4))]
  ##How do I choose the dimensions here?
  ##Do i put all the categorical columns here? 
  deep_columns = [
      tf.contrib.layers.embedding_column(title, dimension=8),
      tf.contrib.layers.embedding_column(house, dimension=8),
      tf.contrib.layers.embedding_column(culture, dimension=8),
      tf.contrib.layers.embedding_column(isAliveNoble, dimension=8),
      tf.contrib.layers.embedding_column(numberDeadRelations,
                                         dimension=8),
      tf.contrib.layers.embedding_column(popularity, dimension=8),
      male,
      spouse,
      isPopular,
      spouse,
      isMarried,
  ]
##From here, reading the code beyond i´ve change anything from the tutorial 
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


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval():
  """Train and evaluate the model."""
  train_file_name, test_file_name = maybe_download()
  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      engine="python")
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      engine="python")

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  df_train[LABEL_COLUMN] = (
      df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
  df_test[LABEL_COLUMN] = (
      df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir)
  m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
