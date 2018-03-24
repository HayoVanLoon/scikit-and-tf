import pandas as pd
import tensorflow as tf
from random import random
import uuid
import sys


# load data
data_file = 'data/3x_plus_9.csv'
if len(sys.argv) > 1:
  data_file = 'data/%s' % sys.argv[1]

df = pd.read_csv(data_file, names=['x', 'y'])
df = df.assign(set=[random() for _ in range(len(df))])

train_df = df[df['set'] <= .8].drop('set', axis=1)
test_df = df[df['set'] > .8].drop('set', axis=1)


# describe model
feature_column = tf.feature_column.numeric_column('x')

est = tf.estimator.LinearRegressor(
  feature_columns=[feature_column],
  model_dir='tmp/linreg-tf%s' % hash(data_file))


# define input function helpers

def create_input_fn(df: pd.DataFrame, train=True):
  df2 = df.copy()
  features, labels = df2, df2.pop('y')
  return tf.estimator.inputs.pandas_input_fn(x=features, y=labels,
                                             num_epochs=100 if train else 1,
                                             shuffle=True,
                                             target_column='y')


def create_pred_fn(xs):
  features = pd.DataFrame.from_dict({'x': xs})
  return tf.estimator.inputs.pandas_input_fn(x=features, shuffle=False)


# train & test
est.train(input_fn=create_input_fn(train_df))

metrics = est.evaluate(input_fn=create_input_fn(test_df, train=False))
print(metrics)


# show some predictions
pred_input = [1, 10, 40]
preds = est.predict(input_fn=create_pred_fn(pred_input))

for i, p in zip(pred_input, preds):
  print('%s -> %s' % (i, p['predictions'][0]))

