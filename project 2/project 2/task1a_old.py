#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
sw = stopwords.words('english')


# read the csv files into pd
amazon_small = pd.read_csv('amazon_small.csv')
google_small = pd.read_csv('google_small.csv')

#Find all possible combinations of two data sets and read into a dataframe
amazon_small['key'] = 1
google_small['key'] = 1
data_small = pd.merge(amazon_small,google_small,on='key').drop('key',axis=1)
data_small.columns = ['idAmazon', 'amazon_name', 'amazon_description', 'amazon_manufacturer', 'amazon_price',
                     'idGoogleBase', 'google_name', 'google_description', 'google_manufacturer', 'google_price']


def get_vectors(*strs):
    # normalise the text by remove stopwords, punctuation from string 
    text = [(lemmatizer.lemmatize(w)).lower() for w in strs if (not w in sw) and (not w in string.punctuation)]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)

def get_price_sim(p1, p2):
    p_similarity = min(p1, p2)/max(p1, p2)
    return p_similarity
    
# replace Null with empty string
data_small["amazon_description"].replace(np.nan,"",inplace = True)
data_small["google_description"].replace(np.nan,"", inplace = True)
data_small["amazon_manufacturer"].replace(np.nan,"",inplace = True)
data_small["google_manufacturer"].replace(np.nan,"", inplace = True)

name_similarity = []
des_similarity = []
manu_similarity = []
price_similarity = []
final_score = []
for i in range(len(data_small)):
    name_sim = 0
    des_sim = 0
    manu_sim = 0
    price_sim = 0
    amazon_name = data_small['amazon_name'][i]
    google_name = data_small['google_name'][i]
    name_sim = get_cosine_sim(amazon_name, google_name)[0][1]
    name_similarity.append(name_sim)
    
    amazon_description = data_small['amazon_description'][i]
    google_description = data_small['google_description'][i]
    if(not(amazon_description == '') and not(google_description == '')):
        des_sim = get_cosine_sim(amazon_description, google_description)[0][1]
    des_similarity.append(des_sim)
    
    amazon_manufacturer = data_small["amazon_manufacturer"][i]
    google_manufacturer = data_small["google_manufacturer"][i]
    if(not(amazon_manufacturer == '') and not(google_manufacturer == '')):
        manu_sim = get_cosine_sim(amazon_manufacturer, google_manufacturer)[0][1]
    manu_similarity.append(manu_sim)
    
    if((data_small['amazon_price'][i] != 0) and (data_small['google_price'][i] != 0)):
        price_sim = get_price_sim(data_small['amazon_price'][i], data_small['google_price'][i])
    price_similarity.append(price_sim)
    
    score = 0.6828*name_sim + 0.1444*des_sim + 0.1729*price_sim
    final_score.append(score)
    
data_small['name_similarity'] = name_similarity
data_small['des_similarity'] = des_similarity
data_small['manu_similarity'] = manu_similarity
data_small['price_similarity'] = price_similarity
data_small['final_score'] = final_score


#Determine the matches
amazon_id = []
google_id = []
threshold = 0.5

for i in range(len(data_small)):
    if data_small['final_score'][i] >= threshold:
        amazon_id.append(data_small["idAmazon"][i])
        google_id.append(data_small["idGoogleBase"][i])
match = pd.DataFrame({'idAmazon': amazon_id, 'idGoogleBase': google_id})
match
match.to_csv('task1a.csv', index=False, sep=',')


# find precision and recall
truth = pd.read_csv('amazon_google_truth_small.csv')
t = len(truth)
tp = 0
fn = 0
fp = 0
for i in range(len(match)):
    matchpair = 0
    for j in range(len(truth)):
        
        if match["idAmazon"][i] == truth["idAmazon"][j] and match['idGoogleBase'][i] == truth['idGoogleBase'][j]:
            tp += 1
            matchpair = 1
    if not matchpair:
            fp += 1
fn = t - tp
print(tp)
print(fp)
print(fn)

precision = tp/(fp+tp)
print("precision: {}".format(precision))
recall = tp/(tp+fn)
print('recall: {}'.format(recall))

