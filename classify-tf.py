import pandas as pd
import tensorflow as tf


# load data
train_file = 'data/iris_training.csv'
test_file = 'data/iris_test.csv'

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 
                    'Species']

train_df = pd.read_csv(train_file, names=CSV_COLUMN_NAMES, header=0)
test_df = pd.read_csv(test_file, names=CSV_COLUMN_NAMES, header=0)


# describe model
feature_columns = []
for key in CSV_COLUMN_NAMES[0:-1]:
  feature_columns.append(tf.feature_column.numeric_column(key=key))

est = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                 hidden_units=[10, 10],
                                 n_classes=3,
                                 model_dir='tmp/classify-tf')


# define input function helpers 

def create_input_fn(df: pd.DataFrame, train=True):
  df2 = df.copy()
  features, labels = df2, df2.pop('Species')
  return tf.estimator.inputs.pandas_input_fn(x=features, y=labels,
                                             num_epochs=1000 if train else 1,
                                             shuffle=True,
                                             target_column='Species')


def create_pred_fn(df: pd.DataFrame):
  features  = df.copy().drop('Species', axis=1)
  return tf.estimator.inputs.pandas_input_fn(x=features, shuffle=False)


# train & test 
est.train(input_fn=create_input_fn(train_df))

metrics = est.evaluate(input_fn=create_input_fn(test_df, train=False))
print(metrics)


# just use test set to visualise some predictions
preds = est.predict(input_fn=create_pred_fn(test_df))

misses = 0
for i, p in zip(test_df.Species, preds):
  p2 = p['classes'][0].decode('utf-8') 
  if p2 != str(i):
    misses += 1
    print('(!) %s != %s' % (p2, i)) 

# should match evaluate output
print('simple accuracy: %s' % ((len(test_df) - misses) / len(test_df)))

