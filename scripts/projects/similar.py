import pandas as pd
import numpy as np

# Importing Data
df = pd.read_csv('../../data/projects.csv')

# Converting to Lists
df['tagline']=df['tagline'].apply(lambda x:x.split())
df['description']=df['description'].apply(lambda x:x.split())
df['category']=df['category'].apply(lambda x:x.split())

import ast

def parse(obj):
    try:
        obj = ast.literal_eval(obj)
        return obj
    except:
        return obj

df['tags']=df['tags'].apply(parse)

df['keys']=df['tagline']+df['description']+df['tags']+df['category']

# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download("punkt")

# Stemming
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

ps=PorterStemmer()

def stem(x):
    L = []

    for i in x:
        i=i.lower()
        if i not in L and i not in stopwords.words("english") and i not in string.punctuation:
            L.append(ps.stem(i.lower()))
    return " ".join(L)

df.loc[:,'keys']=df['keys'].apply(stem)

df=df[['id','title','keys']]

# Calculating Similarities
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)
vectors = cv.fit_transform(df['keys']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(vectors)

# Saving the Similarities
import pickle

with open('../../models/projects/similarities.pickle', 'wb') as f:
    pickle.dump(similarities, f)