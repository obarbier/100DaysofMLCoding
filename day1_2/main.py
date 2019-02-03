import numpy as np
import pandas as pd
data = pd.read_csv("data.csv")
"""
   Country   Age   Salary Purchased
0   France  44.0  72000.0        No
1    Spain  27.0  48000.0       Yes
2  Germany  30.0  54000.0        No
3    Spain  38.0  61000.0        No
4  Germany  40.0      NaN       Yes
5   France  35.0  58000.0       Yes
6    Spain   NaN  52000.0        No
7   France  48.0  79000.0       Yes
8  Germany  50.0  83000.0        No
9   France  37.0  67000.0       Yes
"""
X = data.iloc[ : , :-1].values
"""
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 40.0 nan]
 ['France' 35.0 58000.0]
 ['Spain' nan 52000.0]
 ['France' 48.0 79000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]]
"""
Y = data.iloc[ : , 3].values
"""
['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # replace missing value with mean
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
"""
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 40.0 63777.77777777778]
 ['France' 35.0 58000.0]
 ['Spain' 38.77777777777778 52000.0]
 ['France' 48.0 79000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]]
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
# Creating a dummy variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
preprocess = make_column_transformer(
    ( OneHotEncoder(categories="auto"),[0]), # encoding of categorical data
    (StandardScaler(),[1,2]) # feature scale
)
X = preprocess.fit_transform(X)
"""
[[ 1.00000000e+00  0.00000000e+00  0.00000000e+00  7.58874362e-01
   7.49473254e-01]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00 -1.71150388e+00
  -1.43817841e+00]
 [ 0.00000000e+00  1.00000000e+00  0.00000000e+00 -1.27555478e+00
  -8.91265492e-01]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00 -1.13023841e-01
  -2.53200424e-01]
 [ 0.00000000e+00  1.00000000e+00  0.00000000e+00  1.77608893e-01
   6.63219199e-16]
 [ 1.00000000e+00  0.00000000e+00  0.00000000e+00 -5.48972942e-01
  -5.26656882e-01]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
  -1.07356980e+00]
 [ 1.00000000e+00  0.00000000e+00  0.00000000e+00  1.34013983e+00
   1.38753832e+00]
 [ 0.00000000e+00  1.00000000e+00  0.00000000e+00  1.63077256e+00
   1.75214693e+00]
 [ 1.00000000e+00  0.00000000e+00  0.00000000e+00 -2.58340208e-01
   2.93712492e-01]]
"""
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
"""
[0 1 0 0 1 1 0 1 0 1]
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
"""
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00  1.77608893e-01
   6.63219199e-16]
 [ 1.00000000e+00  0.00000000e+00  0.00000000e+00 -2.58340208e-01
   2.93712492e-01]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00 -1.71150388e+00
  -1.43817841e+00]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
  -1.07356980e+00]
 [ 1.00000000e+00  0.00000000e+00  0.00000000e+00  1.34013983e+00
   1.38753832e+00]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00 -1.13023841e-01
  -2.53200424e-01]
 [ 1.00000000e+00  0.00000000e+00  0.00000000e+00  7.58874362e-01
   7.49473254e-01]
 [ 1.00000000e+00  0.00000000e+00  0.00000000e+00 -5.48972942e-01
  -5.26656882e-01]] [[ 0.          1.          0.         -1.27555478 -0.89126549]
 [ 0.          1.          0.          1.63077256  1.75214693]] [1 1 1 0 1 0 0 1] [0 0]
"""
