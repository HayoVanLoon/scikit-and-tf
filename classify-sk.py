import pandas as pd
from sklearn.neural_network import MLPClassifier


# load model
train_file = 'data/iris_training.csv'
test_file = 'data/iris_test.csv'

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 
                    'Species']

train_df = pd.read_csv(train_file, names=CSV_COLUMN_NAMES, header=0)
test_df = pd.read_csv(test_file, names=CSV_COLUMN_NAMES, header=0)


# init model
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)


# train & test
features, labels = train_df, train_df.pop('Species')
model.fit(features, labels)

fs, l = test_df, test_df.pop('Species')
metrics = model.score(fs, l)
print(metrics)


# just use test set to visualise some predictions
preds = model.predict(fs)

misses = 0
for l, p in zip(l, preds):
  if p != l:
    print('(!) %s != %s' % (p, l))
    misses += 1

# should match score output
print('simple accuracy: %s' % ((len(fs) - misses) / len(fs)))

