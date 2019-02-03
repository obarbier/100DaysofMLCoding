# Data Preprocessing
## Goal and Objectives
### Tutorial
  * Numpy (iloc, array calculation) + Pandas
    * https://docs.scipy.org/doc/numpy/user/quickstart.html
  * sklearn (Impute and Preprocessing): Follow Tutorial on Getting Started
    * fit vs fit_transform vs transform
      * https://stackoverflow.com/questions/23838056/what-is-the-difference-between-transform-and-fit-transform-in-sklearn
### Importing Dataset
### Handling Missing Dataset
   Imputer: The Imputer fills missing values with some statistics (e.g. mean, median, ...) of the data. To avoid data leakage during cross-validation, it computes the statistic on the train data during the fit, stores it and uses it on the test data, during the transform.
### Encoding Categorical Dataset
  OneHotEncoder : https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f
  We use one hot encoder to perform “binarization” of the category and include it as a feature to train the model.
  Suppose you have ‘flower’ feature which can take values ‘daffodil’, ‘lily’, and ‘rose’. One hot encoding converts ‘flower’ feature to three features, ‘is_daffodil’, ‘is_lily’, and ‘is_rose’ which all are binary.
### Splitting Dataset into test Set and Training Set
### Feature Scalling
  The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1. Given the distribution of the data, each value in the dataset will have the sample mean value subtracted, and then divided by the standard deviation of the whole dataset.
