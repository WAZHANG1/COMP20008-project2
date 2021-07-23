#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

life = pd.read_csv('life.csv')
world = pd.read_csv('world.csv')
world = world.iloc[:-5]

life_expectancy = life[['Country Code', 'Life expectancy at birth (years)']]
#data = pd.merge(world, life_expectancy, on='Country Code', how='left')
data = pd.merge(world, life_expectancy, on='Country Code')
data.drop('Country Name', axis=1, inplace=True)
data.drop('Time', axis=1, inplace=True)
#data['Life expectancy at birth (years)'].replace(np.nan,'Medium',inplace = True)


medium_list = []
for column in data.columns:
    if column == 'Country Code':
        data[column] = data[column]

    elif column == 'Life expectancy at birth (years)':
        data[column].replace('Low', 0, inplace = True)
        data[column].replace('Medium', 1, inplace = True)
        data[column].replace('High', 2, inplace = True)
    else:
        data[column].replace(['..','...'], np.nan, inplace = True)
        medium = data[column].median()
        medium_list.append(medium)
        data[column].replace(np.nan, medium, inplace = True)
        
feature = data[data.columns[1:21]].astype(float)
np.set_printoptions(suppress=True)
mean_list = feature.describe().loc['mean']
mean = np.asarray(mean_list, dtype=np.float32)
std_list = feature.describe().loc['std']
std = np.asarray(std_list, dtype=np.float32)
var = std*std

# get the features
feature_name = list(feature.columns.values)
data_describe = pd.DataFrame({'feature': feature_name, 'median': medium_list, 'mean': mean, 'variance': var})
pd.set_option('display.float_format', lambda x: '%.3f' % x)
data_describe.to_csv('task2a.csv', index=False, sep=',')


# get just the class labels
classlabel=data['Life expectancy at birth (years)']

# randomly select 2/3 of the instances to be training and the rest to be testing
X_train, X_test, y_train, y_test = train_test_split(feature, classlabel, train_size=0.66, test_size=0.34, random_state=100)

# normalise the data to have 0 mean and unit variance using the library functions.  This will help for later
# computation of distances between instances
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

# decision tree
dt = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=4)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
print('Accuracy of decision tree: {:.3f}'.format(accuracy_score(y_test, y_pred)))

# knn = 5
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print('Accuracy of k-nn (k=5): {:.3f}'.format(accuracy_score(y_test, y_pred)))

# knn = 10
knn1 = neighbors.KNeighborsClassifier(n_neighbors=10)
knn1.fit(X_train, y_train)
y_pred1=knn1.predict(X_test)
print('Accuracy of k-nn (k=10): {:.3f}'.format(accuracy_score(y_test, y_pred1)))


# In[ ]:




