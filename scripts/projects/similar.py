import pandas as pd
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
import ssl
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import traceback

max_threads = 4


def logger(level, title, description, path):
    utils_path = os.path.join(os.getcwd(), "utils")
    subprocess.run(
        ["python3", "api_logger.py", level, title, description, path], cwd=utils_path
    )


try:
    # Importing Data
    csv_path = "data/projects.csv"
    df = pd.read_csv(csv_path)

    # Converting to Lists
    df["tagline"] = df["tagline"].apply(lambda x: x.split())
    df["description"] = df["description"].apply(lambda x: x.split())
    df["category"] = df["category"].apply(lambda x: x.split())

    import ast

    def parse(obj):
        try:
            obj = ast.literal_eval(obj)
            return obj
        except:
            return obj

    df["tags"] = df["tags"].apply(parse)
    df["keys"] = df["tagline"] + df["description"] + df["tags"] + df["category"]

    # Setup NLTK
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
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    punctuation_set = set(string.punctuation)
    custom_stopwords = ["need", "want", "this", "that", "fast"]

    def stem(x):
        L = []
        tagged_tokens = pos_tag(x)
        for token, pos in tagged_tokens:
            token = token.lower()
            if (
                pos not in {"JJ", "JJR", "JJS"}  # Adjective tags
                and token not in custom_stopwords
                and token not in stop_words
                and token not in punctuation_set
            ):
                stemmed_token = ps.stem(token)
                if stemmed_token not in L:
                    L.append(stemmed_token)
        return " ".join(L)

    # Parallelized Stemming
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        keys = list(executor.map(stem, df["keys"].tolist()))

    df["keys"] = keys
    df = df[["id", "title", "keys"]]

    # Calculating Similarities
    cv = TfidfVectorizer(max_features=5000)
    vectors = cv.fit_transform(df["keys"]).toarray()

    similarities = cosine_similarity(vectors)

    # Ensure directory exists before saving
    os.makedirs("models/projects", exist_ok=True)
    with open("models/projects/similarities.pickle", "wb") as f:
        pickle.dump(similarities, f)

    os.remove(csv_path)

    print("------------- Successfully Trained Similar Projects -------------")

    logger(
        "info",
        "Training Successful",
        "Successfully Trained Similar Projects",
        "scripts/projects/similar.py",
    )

except Exception as e:
    error_message = f"Error: {str(e)} \n Traceback: {traceback.format_exc()}"
    print(error_message)

    logger("error", "Training Failed", error_message, "scripts/projects/similar.py")
