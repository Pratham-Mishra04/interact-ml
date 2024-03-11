import pandas as pd
import subprocess

def logger(level , title, description, path):
    subprocess.run(['python3', 'api_logger.py', level, title, description, path], cwd='utils')

try :
    # Importing Data
    df = pd.read_csv('data/openings.csv')

    # Converting to Lists
    df['title']=df['title'].apply(lambda x:x.split())
    df['description']=df['description'].apply(lambda x:x.split())
    df['project_id']=df['project_id'].apply(lambda x:[x])

    import ast

    def parse(obj):
        try:
            obj = ast.literal_eval(obj)
            return obj
        except:
            return obj

    df['tags']=df['tags'].apply(parse)

    df['keys']=df['title']+df['description']+df['tags']+df['project_id']

    import ssl
    import nltk
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")

    # Stemming
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    from nltk import pos_tag, word_tokenize
    import string

    ps=PorterStemmer()

    custom_stopwords = ["need", "want", "this", "that", "fast"]

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

    with open('models/openings/similarities.pickle', 'wb') as f:
        pickle.dump(similarities, f)

    logger("info",f"Training Successful", "Successfully Trained Similar Openings", "scripts/openings/similar.py")
except Exception as e :
    logger("error",f"Training Failed", str(e), "scripts/openings/similar.py")
