# Simple ML Models with Scikit and Tensorflow

This repository illustrates the parallel usage of Scikit-learn and Tensorflow
for creating machine learning models for two simple problems.

One is a simple linear regression on a dataset with data points following ```
y = ax + b + E```. The other is a classification on the Iris data set.

Note that this is not an exercise in deciding which framework is better. The 
aim is to show the similarities in the model creation process.

## Installation
Clone the repo:
```
git clone https://bitbucket.org/incentro-ondemand/scikit-and-tf
```

Create a virtual environment in the repo:
```
virtualenv -p python3.6 venv
```

Activate the new virtual environment:
```
source venv/bin/activate
``` 

Install dependencies:
```
pip install -r requirements.txt
```

All done! The scripts should now run:
```
python linreg-sk.py
python linreg-tf.py
python classify-sk.py
python classify-tf.py
```

## Note on temporary files
While operating, Tensorflow stores (temporary) model data in model directories.
For these models, a tmp folder (with subfolders) will be created in the repo.
While playing around, you might want to delete these as Tensorflow does 
maintain state through these, which might lead to unexpected results.
