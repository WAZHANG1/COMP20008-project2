#!/usr/bin/env python
# coding: utf-8

# In[19]:


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

# combine the informations in world and life, drop the rows without life expectancy 
life_expectancy = life[['Country Code', 'Life expectancy at birth (years)']]
data = pd.merge(world, life_expectancy, on='Country Code')
data.drop('Country Name', axis=1, inplace=True)
data.drop('Time', axis=1, inplace=True)


# In[20]:

# get the features
feature = data[data.columns[1:21]]
# get just the class labels
classlabel=data['Life expectancy at birth (years)']

# randomly select 2/3 of the instances to be training and the rest to be testing
X_train_, X_test_, y_train, y_test = train_test_split(feature, classlabel, train_size=2/3, test_size=1/3, random_state=100)

X_train = pd.DataFrame(X_train_)

# impute the median of the training set and replace the missing values with its median
medium_list = []
for column in X_train.columns:
    X_train[column].replace('..', np.nan, inplace = True)
    medium = X_train[column].median()
    medium_list.append(medium)
    X_train[column].replace(np.nan, medium, inplace = True)

X_train = X_train[X_train.columns[:]].astype(float)
#X_train.describe()


# In[21]:

# calculate the mean and variance of the training set
np.set_printoptions(suppress=True)
median = np.asarray(medium_list, dtype=np.float32)
mean_list = X_train.describe().loc['mean']
mean = np.asarray(mean_list, dtype=np.float32)
std_list = X_train.describe().loc['std']
std = np.asarray(std_list, dtype=np.float32)
var = std*std

feature_name = list(feature.columns.values)
data_describe = pd.DataFrame({'feature': feature_name, 'median': np.around(median, 3), 'mean': np.around(mean, 3), 'variance': np.around(var, 3)})
pd.set_option('display.float_format', lambda x: '%.3f' % x)
data_describe.to_csv('task2a.csv', index=False, sep=',')
#data_describe


# In[22]:

# fill the missing value in test set with test set's median
X_test = pd.DataFrame(X_test_)
for column in X_test.columns:
    X_test[column].replace('..', np.nan, inplace = True)
    medium = X_test[column].median()
    X_test[column].replace(np.nan, medium, inplace = True)

# normalise the data to have 0 mean and unit variance using the library functions.  This will help for later
# computation of distances between instances
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# In[23]:


# decision tree
dt = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=4)
dt.fit(X_train, y_train)
y_pred1=dt.predict(X_test)
print('Accuracy of decision tree: {:.3f}%'.format(100 * accuracy_score(y_test, y_pred1)))

# knn = 5
knn1 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train, y_train)
y_pred2=knn1.predict(X_test)
print('Accuracy of k-nn (k=5): {:.3f}%'.format(100 * accuracy_score(y_test, y_pred2)))

# knn = 10
knn2 = neighbors.KNeighborsClassifier(n_neighbors=10)
knn2.fit(X_train, y_train)
y_pred3=knn2.predict(X_test)
print('Accuracy of k-nn (k=10): {:.3f}%'.format(100 * accuracy_score(y_test, y_pred3)))


# In[ ]:




