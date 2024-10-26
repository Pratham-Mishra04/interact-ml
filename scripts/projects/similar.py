import pandas as pd
import numpy as np
import subprocess
import ssl
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import ast
from concurrent.futures import ProcessPoolExecutor

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

ps=PorterStemmer()

custom_stopwords = ["need", "want", "this", "that", "fast"]

def logger(level , title, description, path):
    subprocess.run(['python3', 'api_logger.py', level, title, description, path], cwd='utils')

#Stemming
def stem(x):
    L = []
    tagged_tokens = pos_tag(x)
    for token, pos in tagged_tokens:
        token=token.lower()
        if pos != 'JJ' and pos != 'JJR' and pos != 'JJS' and token not in custom_stopwords:  # Remove adjectives
            stemmed_token = ps.stem(token)
            if stemmed_token not in L and stemmed_token not in stopwords.words("english") and stemmed_token not in string.punctuation:
                L.append(stemmed_token)
    return " ".join(L)

def parse(obj):
        try:
            obj = ast.literal_eval(obj)
            return obj
        except:
            return obj

def vectorize_chunk(texts, max_features=5000):
    cv = CountVectorizer(max_features=max_features)
    return cv.fit_transform(texts).toarray(), cv

def cosine_similarity_chunk(start_idx, vectors_chunk, full_vectors):
    return cosine_similarity(vectors_chunk, full_vectors)

try:
    # Importing Data
    df = pd.read_csv('data/projects.csv')

    # Converting to Lists
    df['tagline']=df['tagline'].apply(lambda x:x.split())
    df['description']=df['description'].apply(lambda x:x.split())
    df['category']=df['category'].apply(lambda x:x.split())

    df['tags']=df['tags'].apply(parse)

    df['keys']=df['tagline']+df['description']+df['tags']+df['category']
    
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    df.loc[:,'keys']=df['keys'].apply(stem)
    df=df[['id','title','keys']]

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

    vectors = np.vstack(vectors_list)

    #Calculating similaries in chunks
    similarities_list = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(cosine_similarity_chunk, i, vectors[i:i + chunk_size], vectors)
                for i in range(0, len(vectors), chunk_size)]
        for future in futures:
            similarities_list.append(future.result())

    #Combining all chunks
    similarities = np.vstack(similarities_list)

    # Saving the Similarities
    with open('models/projects/similarities.pickle', 'wb') as f:
        pickle.dump(similarities, f)

    logger("info",f"Training Successful", "Successfully Trained Similar Projects", "scripts/projects/similar.py")
except Exception as e :
    logger("error",f"Training Failed", str(e), "scripts/projects/similar.py")
