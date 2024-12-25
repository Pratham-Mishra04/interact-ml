from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import traceback

max_threads = 4


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Load model from HuggingFace Hub
miniLM_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
miniLM_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

from functools import lru_cache


@lru_cache(maxsize=None)
def get_embeddings(sentences, model, tokenizer):
    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


from sklearn.metrics.pairwise import cosine_similarity


def get_similarity_score(emb1, emb2):
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)

    # Calculate cosine similarity
    return cosine_similarity(emb1, emb2)[0][0]


initial_topics_scores = {
    "Design": 0,
    "Web Development": 0,
    "Machine Learning": 0,
    "UI/UX": 0,
    "Management": 0,
    "Finance": 0,
    "Recruitments": 0,
    "Cyber security": 0,
    "Video Editing": 0,
    "Content Writing": 0,
    "DSA": 0,
    "Dev Ops": 0,
    "Photography": 0,
}


def preprocess_weights(config):
    zero_keys = []
    for key in config.keys():
        if len(config[key]["tags"]) == 0:
            config[key]["weight"] = 0
            zero_keys.append(key)

    if len(zero_keys) > 0:
        constant_keys = ["user"]
        remaining_keys = [
            key
            for key in config.keys()
            if key not in constant_keys and key not in zero_keys
        ]

        if len(remaining_keys) == 0:
            split_weight = 1 / len(constant_keys)
            for key in constant_keys:
                config[key]["weight"] = split_weight
            return config

        remaining_weight = 1
        for key in constant_keys:
            remaining_weight -= config[key]["weight"]

        split_weight = remaining_weight / len(remaining_keys)
        if split_weight >= 0.25:
            split_weight = 1 / len(constant_keys + remaining_keys)
            for key in constant_keys + remaining_keys:
                config[key]["weight"] = split_weight
        else:
            for key in remaining_keys:
                config[key]["weight"] = split_weight

    return config


import numpy as np


def assign_weights(n, equal=False):
    if equal:
        return [1 / n for i in range(n)]

    weights = np.linspace(n, 1, num=n)

    # Normalize the weights so that their sum is 1
    normalized_weights = weights / np.sum(weights)

    return normalized_weights


def assign_tag_weights(config):
    equal_keys = ["followings", "member_organisations"]

    for key in config.keys():
        if config[key]["type"] == "multiple" and len(config[key]["tags"]) > 0:
            is_equal = False
            if key in equal_keys:
                is_equal = True
            config[key]["tag_weights"] = assign_weights(
                len(config[key]["tags"]), equal=is_equal
            )

    return config


from concurrent.futures import ThreadPoolExecutor


def increment_score(tags, weight, scores_obj, threshold=0.3):
    if tags is not None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda tag: compute_score(tag, scores_obj, weight, threshold), tags
                )
            )
        # Combine the scores from all parallel tasks
        for updated_scores in results:
            for topic, score in updated_scores.items():
                scores_obj[topic] += score
    return scores_obj


def compute_score(tag, scores_obj, weight, threshold):
    """Compute the similarity score for a single tag."""
    temp_scores = {key: 0 for key in scores_obj.keys()}
    for topic in scores_obj.keys():
        topic_emb = get_embeddings(topic, miniLM_model, miniLM_tokenizer)
        tag_emb = get_embeddings(tag, miniLM_model, miniLM_tokenizer)
        score = get_similarity_score(topic_emb, tag_emb)
        if score > threshold:
            temp_scores[topic] += score * weight
    return temp_scores


import copy


def get_recommended_topics(config, limit=5):
    topics_scores = copy.deepcopy(initial_topics_scores)

    config = assign_tag_weights(preprocess_weights(config))

    for key in config.keys():
        if config[key]["type"] == "multiple" and len(config[key]["tags"]) > 0:
            for tags, tag_weight in zip(
                config[key]["tags"], config[key]["tag_weights"]
            ):
                topics_scores = increment_score(
                    tags, config[key]["weight"] * tag_weight, topics_scores
                )
        else:
            topics_scores = increment_score(
                config[key]["tags"], config[key]["weight"], topics_scores
            )

    top_topics = sorted(topics_scores.items(), key=lambda item: item[1], reverse=True)[
        :limit
    ]
    return top_topics


from psycopg2 import sql


def batch_update_user_topics(conn, user_topics):
    cursor = conn.cursor()
    for user_id, topics in user_topics:
        topics_str = "{" + ",".join(f'"{topic}"' for topic in topics) + "}"
        update_query = sql.SQL("UPDATE users SET topics = %s WHERE id = %s")
        cursor.execute(update_query, (topics_str, user_id))
    conn.commit()
    cursor.close()


import numpy as np
import json
import subprocess
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()


def logger(level, title, description, path):
    utils_path = os.path.join(os.getcwd(), "utils")

    subprocess.run(
        ["python3", "api_logger.py", level, title, description, path], cwd=utils_path
    )


try:
    conn = psycopg2.connect(
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )

    json_path = "data/topics.json"
    with open(json_path, "r") as f:
        configs = json.load(f)

    from concurrent.futures import ThreadPoolExecutor

    def process_user_config(user_config):
        """Process a single user's configuration to get recommended topics."""
        results = []
        for user_id, config in user_config.items():
            topic_scores = get_recommended_topics(config, 4)
            topics = [topics for topics, score in topic_scores]
            results.append((user_id, topics))
        return results

    # Parallelize processing of user configurations
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        all_user_topics = list(executor.map(process_user_config, configs))

    flattened_user_topics = [item for sublist in all_user_topics for item in sublist]
    batch_update_user_topics(conn, flattened_user_topics)

    conn.close()

    os.remove(json_path)

    print("------------- Successfully Trained Topics for Users -------------")

    logger(
        "info",
        f"Training Successful",
        "Successfully Trained User Recommended Topics",
        "scripts/topics.py",
    )
except Exception as e:
    error_message = f"Error: {str(e)} \n Traceback: {traceback.format_exc()}"
    print(error_message)

    logger("error", f"Training Failed", error_message, "scripts/topics.py")
