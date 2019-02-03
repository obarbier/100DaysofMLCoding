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
from sklearn.compose import ColumnTransformer, make_column_transformer
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
preprocess = make_column_transformer(
    ( OneHotEncoder(categories='auto'),[0])
)
X = preprocess.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
print("Splitting the datasets into training sets and Test sets")
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
print("Feature Scaling")
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_train)
