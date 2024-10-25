import pandas as pd
import subprocess
from concurrent.futures import ProcessPoolExecutor
import ast
import ssl
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

ps=PorterStemmer()

custom_stopwords = ["need", "want", "this", "that", "fast"]

def logger(level , title, description, path):
    subprocess.run(['python3', 'api_logger.py', level, title, description, path], cwd='utils')

def parse(obj):
        try:
            obj = ast.literal_eval(obj)
            return obj
        except:
            return obj

#Stemming
def stem(x):
    if not isinstance(x, list):
        return []
    
    L = []
    for token in x:
        if isinstance(token, str):
            tagged_token = pos_tag([token])
            token = token.lower()
            pos = tagged_token[0][1]
            if pos not in {'JJ', 'JJR', 'JJS'} and token not in custom_stopwords:  # Remove adjectives
                stemmed_token = ps.stem(token)
                if stemmed_token not in L and stemmed_token not in stopwords.words("english") and stemmed_token not in string.punctuation:
                    L.append(stemmed_token)
    return ' '.join(L)

# Vectorize chunks of data
def vectorize_chunk(texts, max_features=5000):
    cv = CountVectorizer(max_features=max_features)
    return cv.fit_transform(texts).toarray(), cv

#Calculate similarities
def cosine_similarity_chunk(start_idx, vectors_chunk, full_vectors):
    return cosine_similarity(vectors_chunk, full_vectors)

try :
    # Importing Data
    df = pd.read_csv('data/openings.csv')

    # Converting to Lists
    df['title']=df['title'].apply(lambda x:x.split())
    df['description']=df['description'].apply(lambda x:x.split())
    df['project_id']=df['project_id'].apply(lambda x:[x])

    df['tags']=df['tags'].apply(parse)

    df['keys']=df['title']+df['description']+df['tags']+df['project_id']

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    df.loc[:,'keys']=df['keys'].apply(stem)
    df = df[['id','title','keys']]
    n_chunks = 4
    chunk_size = len(df) // n_chunks
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    vectors_list = []
    count_vectorizers = []

    #Vectorizing in chunks
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(vectorize_chunk, chunk['keys'].apply(lambda x: ' '.join(x)))
                for chunk in chunks]
        for future in futures:
            vectors_chunk, cv = future.result()
            vectors_list.append(vectors_chunk)
            count_vectorizers.append(cv)

    #Combining vectors
    vectors = np.vstack(vectors_list)

    #Calculating similarities in chunks
    similarities_list = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(cosine_similarity_chunk, i, vectors[i:i + chunk_size], vectors)
                for i in range(0, len(vectors), chunk_size)]
        for future in futures:
            similarities_list.append(future.result())

    #Combining similarities
    similarities = np.vstack(similarities_list)

    # Saving the Similarities
    with open('models/openings/similarities.pickle', 'wb') as f:
        pickle.dump(similarities, f)

    logger("info",f"Training Successful", "Successfully Trained Similar Openings", "scripts/openings/similar.py")
except Exception as e :
    logger("error",f"Training Failed", str(e), "scripts/openings/similar.py")
