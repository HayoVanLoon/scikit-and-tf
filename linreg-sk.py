import pandas as pd
from sklearn import linear_model
from random import random
import sys

# load data
data_file = 'data/3x_plus_9.csv'
if len(sys.argv) > 1:
  data_file = 'data/%s' % sys.argv[1]
 
df = pd.read_csv(data_file, names=['features', 'labels'])
df = df.assign(set=[random() for _ in range(len(df))])

train_df = df[df['set'] <= .8].drop('set', axis=1)
test_df = df[df['set'] > .8].drop('set', axis=1)


# init model
lm = linear_model.LinearRegression()


# train & test
lm.fit(train_df.features.values.reshape(-1, 1), train_df['labels'])
print('coef: %s' % lm.coef_)
print('intercept: %s' % lm.intercept_)

metrics = lm.score(test_df.features.values.reshape(-1, 1), test_df['labels'])
print(metrics)


# do some predictions
for x in [1, 10, 40]:
  print('%s -> %s' % (x, lm.predict(x)))

