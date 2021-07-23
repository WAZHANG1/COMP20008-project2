#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


# In[26]:


life = pd.read_csv('life.csv')
world = pd.read_csv('world.csv')
world = world.iloc[:-5]

# combine the informations in world and life, drop the rows without life expectancy 
life_expectancy = life[['Country Code', 'Life expectancy at birth (years)']]
data = pd.merge(world, life_expectancy, on='Country Code')
data.drop('Country Name', axis=1, inplace=True)
data.drop('Time', axis=1, inplace=True)


# In[27]:


# fill every missing values with its column median
for column in data.columns:
    if column == 'Country Code' or column == 'Life expectancy at birth (years)':
        data[column] = data[column]
    else:
        data[column].replace('..', np.nan, inplace = True)
        medium = data[column].median()
        data[column].replace(np.nan, medium, inplace = True)

# get the features
feature = data[data.columns[1:21]].astype(float)
# get just the class labels
classlabel = data['Life expectancy at birth (years)'].values
#data


# In[28]:


# determine the # of k cluster by finding the silhouette coefficient
score = []  
X = feature

for i in range(1, 11):
    model = KMeans(n_clusters=i+1)
    model.fit(X)
    score.append(silhouette_score(X, model.labels_))

fig = plt.figure()
plt.plot(range(2, 12), score)
fig.suptitle('number of cluster vs performance')
plt.xlabel('number of clusters')
plt.ylabel('silhouette score')
plt.savefig('task2bgraph1.png')


# In[29]:

# find the cluster labels
model = KMeans(n_clusters=3)
model.fit(feature)
cluster = pd.DataFrame({'cluster': model.predict(feature)})
feature_data = pd.concat([feature, cluster], axis=1)
feature_data.reset_index(drop=True, inplace=True)
#feature_data


# In[30]:


# make a dataframe with original 20 features, cluster label and interaction pairs with the 20 features
for i in range(len(feature.columns)):
    for j in range(i+1, 20):
        interaction = ('{} x {}'.format(feature.columns[i], feature.columns[j]))
        feature_data[interaction] = feature.iloc[:, i].astype(float) * feature.iloc[:, j].astype(float)
#feature_data


# In[31]:


# randomly select 2/3 of the instances to be training and the rest to be testing
X_train1, X_test1, y_train1, y_test1 = train_test_split(feature_data, classlabel, train_size=2/3, test_size=1/3, random_state=100)


# In[32]:


# calculate the NMI for each feature
X_train1 = pd.DataFrame(X_train1)
columns_NMI = ['feature', 'NMI']
feature_NMI = pd.DataFrame(columns = columns_NMI)
y = 0
for col in X_train1:
    NMI_list = []
    X_train1[col] = X_train1[col].apply(pd.to_numeric, errors = 'coerce')
    NMI = normalized_mutual_info_score(X_train1[col], y_train1)
    NMI_list.append(col)
    NMI_list.append(NMI)
    feature_NMI.loc[y] = NMI_list
    y += 1
#feature_NMI


# In[33]:

# find the top 4 NMI features
feature_NMI.sort_values(by= 'NMI', ascending = False, inplace = True)
top_feature_name = feature_NMI.iloc[0:4, 0].to_list()
top_feature = pd.DataFrame(columns = top_feature_name)
for col in feature_data:
    if col in top_feature_name:
        top_feature[col] = feature_data[col]
feature_name = top_feature.columns.to_list()


# In[34]:


'''interaction term pair features'''
# normalise the data to have 0 mean and unit variance using the library functions.  This will help for later
# computation of distances between instances

X_train = X_train1[feature_name]
scaler = preprocessing.StandardScaler().fit(X_train1)
X_train=scaler.transform(X_train1)
X_test = X_test1[feature_name]
X_test=scaler.transform(X_test1)

# knn = 5
knn1 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train, y_train1)
y_pred1=knn1.predict(X_test)
print('Accuracy of feature engineering: {:.3f}%'.format(100 * accuracy_score(y_test1, y_pred1)))


# In[35]:


'''PCA'''
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    feature, classlabel, train_size = 2/3, test_size = 1/3, random_state = 100)

# normalise the data to have 0 mean and unit variance using the library functions.  This will help for later
# computation of distances between instances
scaler2 = preprocessing.StandardScaler().fit(X_train2)
X_train2=scaler2.transform(X_train2)
X_test2=scaler2.transform(X_test2)

pca = PCA(n_components=4)
X_train2 = pca.fit_transform(X_train2)
X_test2 = pca.fit_transform(X_test2)

# knn = 5
knn2 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train2, y_train2)
y_pred2=knn2.predict(X_test2)
print('Accuracy of PCA: {:.3f}%'.format(100 * accuracy_score(y_test2, y_pred2)))


# In[36]:


'''first four features'''
# get feature D to G
feature_DG = data[data.columns[1:5]].astype(float)
# get just the class labels

# randomly select 2/3 of the instances to be training and the rest to be testing
X_train3, X_test3, y_train3, y_test3 = train_test_split(feature_DG, classlabel, train_size = 2/3, test_size = 1/3, random_state = 100)

# normalise the data to have 0 mean and unit variance using the library functions.  This will help for later
# computation of distances between instances
scaler3 = preprocessing.StandardScaler().fit(X_train3)
X_train3=scaler3.transform(X_train3)
X_test3=scaler3.transform(X_test3)

# knn = 5
knn3 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn3.fit(X_train3, y_train3)
y_pred3=knn3.predict(X_test3)
print('Accuracy of first four features: {:.3f}%'.format(100 * accuracy_score(y_test3, y_pred3)))


# In[ ]:




