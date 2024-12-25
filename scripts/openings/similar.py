import pandas as pd
import subprocess
import os
import swifter
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import ssl
import nltk
import traceback


def logger(level, title, description, path):
    utils_path = os.path.join(os.getcwd(), "utils")
    subprocess.run(
        ["python3", "api_logger.py", level, title, description, path], cwd=utils_path
    )


try:
    # Importing Data
    csv_path = "data/openings.csv"
    df = pd.read_csv(csv_path)

    # Converting to Lists
    df["title"] = df["title"].apply(lambda x: x.split())
    df["description"] = df["description"].apply(lambda x: x.split())
    df["project_id"] = df["project_id"].apply(lambda x: [x])

    import ast

    def parse(obj):
        try:
            obj = ast.literal_eval(obj)
            return obj
        except:
            return obj

    df["tags"] = df["tags"].apply(parse)
    df["keys"] = df["title"] + df["description"] + df["tags"] + df["project_id"]

    # Set up SSL for NLTK
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng")

    # Stemming
    ps = PorterStemmer()
    custom_stopwords = ["need", "want", "this", "that", "fast"]

    def stem(x):
        if not isinstance(x, list):
            return []

        L = []
        for token in x:
            if isinstance(token, str):
                tagged_token = pos_tag([token])
                token = token.lower()
                pos = tagged_token[0][1]
                if pos not in {"JJ", "JJR", "JJS"} and token not in custom_stopwords:
                    stemmed_token = ps.stem(token)
                    if (
                        stemmed_token not in L
                        and stemmed_token not in stopwords.words("english")
                        and stemmed_token not in string.punctuation
                    ):
                        L.append(stemmed_token)
        return " ".join(L)

    # Parallelized stemming using swifter
    df["keys"] = df["keys"].swifter.apply(stem)
    df = df[["id", "title", "keys"]]

    # Vectorization
    cv = TfidfVectorizer(max_features=5000)
    vectors = cv.fit_transform(df["keys"]).toarray()

    # Similarity Calculation
    similarities = cosine_similarity(vectors)

    # Ensure directory exists before saving
    os.makedirs("models/openings", exist_ok=True)
    # Saving the Similarities
    with open("models/openings/similarities.pickle", "wb") as f:
        pickle.dump(similarities, f)

    print("------------- Successfully Trained Similar Openings -------------")

    logger(
        "info",
        "Training Successful",
        "Successfully Trained Similar Openings",
        "scripts/openings/similar.py",
    )
except Exception as e:
    error_message = f"Error: {str(e)} \n Traceback: {traceback.format_exc()}"
    print(error_message)

    logger("error", "Training Failed", error_message, "scripts/openings/similar.py")
