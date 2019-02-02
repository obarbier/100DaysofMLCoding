print("Importing the required Libraries")
import numpy as np
import pandas as pd
print("importing Data set with read_csv")
data = pd.read_csv("data.csv")
print(data.head(1))
X = data.iloc[ : , :-1].values
Y = data.iloc[ : , 3].values
print(X)
print(Y)
print("handling Missing data")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print(X)
print("Encoding Categorical Data")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print(X)
