#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# In[7]:


amazon = pd.read_csv('amazon.csv')
google = pd.read_csv('google.csv')
truth = pd.read_csv("amazon_google_truth.csv")

# convert gbp to aud in google
for i in range(len(google)):
    if 'gbp' in google["price"][i]:
        google["price"][i] = re.sub(r'(.*?) gbp$', str(float(google["price"][i][:-4]) * 1.86), google["price"][i])
        
google["price"] = google["price"].astype(float)

'''print(amazon['price'].max())
print(google['price'].max())'''

# make a block list by price
block_list = []
for i in range(0, 100, 15):
    block_list.append(i)
for i in range(100, 1000, 100):
    block_list.append(i)
for i in range(1000, 10000, 1000):
    block_list.append(i)
for i in range(10000, 100000, 10000):
    block_list.append(i)
block_list.append(1000000)
#block_list


# In[3]:

# create block key name with the price intervals
block_name = []
for i in range(len(block_list)-1):
    name = '({}, {}]'.format(block_list[i], block_list[i + 1])
    block_name.append(name)

# replace the price with price intervals
google['price'] = pd.cut(google['price'], bins = block_list, labels = block_name)
amazon['price'] = pd.cut(amazon['price'], bins = block_list, labels = block_name)

# make a dictionary with price interval as key and values are another to dictionary amazon and google
block_dict = {}
for key in block_name:
    key_dict = {}
    key_dict['amazon'] = []
    key_dict['google'] = []
    block_dict[key] = key_dict
block_dict[np.nan] = key_dict
#print(block_dict)


# In[4]:


# loop through each product and append the product id according to its price interval
for index, row in amazon.iterrows():
    if row['price'] == np.nan:
        block_dict[np.nan]['amazon'].append(row['idAmazon'])
    else:
        block_dict[row['price']]['amazon'].append(row['idAmazon'])
for index, row in google.iterrows():
    if row['price'] == np.nan:
        block_dict[np.nan]['google'].append(row['id'])
    else:
        block_dict[row['price']]['google'].append(row['id'])
#block_dict      


# In[6]:

# output the results into csv files
amazon_block = pd.DataFrame({'block_key': amazon['price'], 'product_id': amazon['idAmazon']})
google_block = pd.DataFrame({'block_key': google['price'], 'product_id': google['id']})
amazon_block['block_key'].replace(np.nan, 'nan', inplace = True)
amazon_block.to_csv('amazon_blocks.csv', index=False, sep=',')
google_block.to_csv('google_blocks.csv', index=False, sep=',')
amazon_block


# In[ ]:




