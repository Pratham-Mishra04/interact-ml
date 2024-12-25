import pandas as pd
import numpy as np
import json
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
import traceback

max_threads = 4


def logger(level, title, description, path):
    utils_path = os.path.join(os.getcwd(), "utils")

    subprocess.run(
        ["python3", "api_logger.py", level, title, description, path], cwd=utils_path
    )


try:
    # * Model Building
    df = pd.read_csv("data/post_scores.csv")

    from sklearn.preprocessing import LabelEncoder

    user_le = LabelEncoder()
    df["enc_user_id"] = user_le.fit_transform(df["user_id"])

    post_le = LabelEncoder()
    df["enc_post_id"] = post_le.fit_transform(df["post_id"])

    X = df[["enc_user_id", "enc_post_id"]]
    y = df["score"]

    import tensorflow as tf
    from keras.models import Model
    from keras.layers import Input, Dense, Embedding, Flatten, concatenate
    from keras.regularizers import l2
    from keras.callbacks import EarlyStopping

    num_users = df["enc_user_id"].nunique()
    num_posts = df["enc_post_id"].nunique()
    k = 100
    l2_lambda = 0.001

    u_input = Input((1,), name="user_input")
    u = Embedding(num_users, k, name="user_emb")(u_input)
    u = Flatten(name="user_flat")(u)
    u = Dense(48, activation="relu", name="user_dense")(u)

    p_input = Input((1,), name="post_input")
    p = Embedding(num_posts, k, name="post_emb")(p_input)
    p = Flatten(name="post_flat")(p)
    p = Dense(8, activation="relu", name="post_dense")(p)

    x = concatenate([u, p], name="concat")
    x = Dense(16, activation="relu", name="dense1")(x)
    x = Dense(4, activation="relu", name="dense2")(x)

    u_bias = Embedding(
        num_users, 1, embeddings_regularizer=l2(l2_lambda), name="user_bias_emb"
    )(u_input)
    u_bias = Flatten(name="user_bias_flat")(u_bias)

    p_bias = Embedding(
        num_posts, 1, embeddings_regularizer=l2(l2_lambda), name="post_bias_emb"
    )(p_input)
    p_bias = Flatten(name="post_bias_flat")(p_bias)

    o = concatenate([x, u_bias, p_bias], name="combined_features")
    o = Dense(16, activation="relu", name="combined_dense1")(o)
    o = Dense(4, activation="relu", name="combined_dense2")(o)
    o = Dense(1, activation="linear", name="output")(o)

    model = Model(inputs=[u_input, p_input], outputs=o)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(0.001), loss="mean_squared_error"
    )

    early_stopping = EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        x=[X["enc_user_id"], X["enc_post_id"]],
        y=y,
        epochs=200,
        verbose=0,
        callbacks=[early_stopping],
    )

    tf.keras.models.save_model(model, "models/posts/recommendations.h5")

    # * Saving Embeddings
    def get_user_embedding(enc_user_id):
        user_embedding = model.get_layer("user_emb")(np.array([enc_user_id]))
        user_embedding = tf.keras.backend.flatten(user_embedding)
        user_embedding = tf.expand_dims(user_embedding, axis=0)
        user_dense = model.get_layer("user_dense")(user_embedding)
        return user_le.inverse_transform([enc_user_id])[0], user_dense.numpy().tolist()

    def get_post_embedding(enc_post_id):
        post_embedding = model.get_layer("post_emb")(np.array([enc_post_id]))
        post_embedding = tf.keras.backend.flatten(post_embedding)
        post_embedding = tf.expand_dims(post_embedding, axis=0)
        post_dense = model.get_layer("post_dense")(post_embedding)
        return post_le.inverse_transform([enc_post_id])[0], post_dense.numpy().tolist()

    def get_user_bias_embedding(enc_user_id):
        user_bias_embedding = model.get_layer("user_bias_emb")(np.array([enc_user_id]))
        user_bias_embedding = tf.keras.backend.flatten(user_bias_embedding)
        return (
            user_le.inverse_transform([enc_user_id])[0],
            user_bias_embedding.numpy().tolist(),
        )

    def get_post_bias_embedding(enc_post_id):
        post_bias_embedding = model.get_layer("post_bias_emb")(np.array([enc_post_id]))
        post_bias_embedding = tf.keras.backend.flatten(post_bias_embedding)
        return (
            post_le.inverse_transform([enc_post_id])[0],
            post_bias_embedding.numpy().tolist(),
        )

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        user_embeddings = dict(
            executor.map(get_user_embedding, df["enc_user_id"].unique())
        )
        post_embeddings = dict(
            executor.map(get_post_embedding, df["enc_post_id"].unique())
        )
        user_bias_embeddings = dict(
            executor.map(get_user_bias_embedding, df["enc_user_id"].unique())
        )
        post_bias_embeddings = dict(
            executor.map(get_post_bias_embedding, df["enc_post_id"].unique())
        )

    with open("models/posts/user_embeddings.json", "w") as f:
        json.dump(user_embeddings, f)

    with open("models/posts/post_embeddings.json", "w") as f:
        json.dump(post_embeddings, f)

    with open("models/posts/user_bias_embeddings.json", "w") as f:
        json.dump(user_bias_embeddings, f)

    with open("models/posts/post_bias_embeddings.json", "w") as f:
        json.dump(post_bias_embeddings, f)

    logger(
        "info",
        f"Training Successful",
        "Successfully Trained Recommended Posts",
        "scripts/posts/recommendation.py",
    )
except Exception as e:
    error_message = f"Error: {str(e)} \n Traceback: {traceback.format_exc()}"
    logger(
        "error", f"Training Failed", error_message, "scripts/posts/recommendation.py"
    )
