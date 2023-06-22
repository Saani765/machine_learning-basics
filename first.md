```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```


```python
ds=pd.read_csv(r"C:\Users\saani\OneDrive\Desktop\Machine Learning-A-Z-Codes-Datasets\Machine Learning A-Z (Codes and Datasets)\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv")
ds.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Country    10 non-null     object 
     1   Age        9 non-null      float64
     2   Salary     9 non-null      float64
     3   Purchased  10 non-null     object 
    dtypes: float64(2), object(2)
    memory usage: 448.0+ bytes
    


```python
X=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values
print(X)
print(y)
```

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
    ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
    


```python
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print(X)

```

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
    


```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
print(X)
```

    [[1.0 0.0 0.0 44.0 72000.0]
     [0.0 0.0 1.0 27.0 48000.0]
     [0.0 1.0 0.0 30.0 54000.0]
     [0.0 0.0 1.0 38.0 61000.0]
     [0.0 1.0 0.0 40.0 63777.77777777778]
     [1.0 0.0 0.0 35.0 58000.0]
     [0.0 0.0 1.0 38.77777777777778 52000.0]
     [1.0 0.0 0.0 48.0 79000.0]
     [0.0 1.0 0.0 50.0 83000.0]
     [1.0 0.0 0.0 37.0 67000.0]]
    


```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)
```

    [0 1 0 0 1 1 0 1 0 1]
    


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


```


```python
print(X_train)
```

    [[0.0 0.0 1.0 38.77777777777778 52000.0]
     [0.0 1.0 0.0 40.0 63777.77777777778]
     [1.0 0.0 0.0 44.0 72000.0]
     [0.0 0.0 1.0 38.0 61000.0]
     [0.0 0.0 1.0 27.0 48000.0]
     [1.0 0.0 0.0 48.0 79000.0]
     [0.0 1.0 0.0 50.0 83000.0]
     [1.0 0.0 0.0 35.0 58000.0]]
    


```python
print(X_test)
```

    [[0.0 1.0 0.0 30.0 54000.0]
     [1.0 0.0 0.0 37.0 67000.0]]
    


```python
print(y_train)
```

    [0 1 0 0 1 1 0 1]
    


```python
print(y_test)
```

    [0 1]
    


```python
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.fit_transform(X_test[:,3:])


```


```python
print(X_train)
```

    [[0.0 0.0 1.0 -0.19159184384578545 -1.0781259408412425]
     [0.0 1.0 0.0 -0.014117293757057777 -0.07013167641635372]
     [1.0 0.0 0.0 0.566708506533324 0.633562432710455]
     [0.0 0.0 1.0 -0.30453019390224867 -0.30786617274297867]
     [0.0 0.0 1.0 -1.9018011447007988 -1.420463615551582]
     [1.0 0.0 0.0 1.1475343068237058 1.232653363453549]
     [0.0 1.0 0.0 1.4379472069688968 1.5749910381638885]
     [1.0 0.0 0.0 -0.7401495441200351 -0.5646194287757332]]
    


```python
print(X_test)
```

    [[0.0 1.0 0.0 -1.0 -1.0]
     [1.0 0.0 0.0 1.0 1.0]]
    


```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score

model=DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error
print("Mean Squared Error:", mse)
print('r2_score: ', round(r2_score(y_test, np.exp(y_pred))*100, 2))
```

    Mean Squared Error: 0.5
    r2_score:  -1968.31
    


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create an instance of the RandomForestRegressor model
model = RandomForestRegressor()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error
print("Mean Squared Error:", mse)
print('r2_score: ', round(r2_score(y_test, np.exp(y_pred)*100), 2))

```

    Mean Squared Error: 0.28419999999999995
    r2_score:  -164697.14
    
