import pandas as pd
import numpy as np
import json
import subprocess

def logger(level , title, description, path):
    subprocess.run(['python3', 'api_logger.py', level, title, description, path], cwd='utils')

try:
    #* Model Building
    df = pd.read_csv('data/post_scores.csv')

    from sklearn.preprocessing import LabelEncoder

    user_le = LabelEncoder()
    df['enc_user_id'] = user_le.fit_transform(df['user_id'])

    post_le = LabelEncoder()
    df['enc_post_id'] = post_le.fit_transform(df['post_id'])

    X = df[['enc_user_id', 'enc_post_id']]
    y = df['score']

    import tensorflow as tf
    from keras.models import Model
    from keras.layers import Input,Dense,Embedding,Flatten,Input,concatenate
    from keras.regularizers import l2

    num_users = df['enc_user_id'].nunique()
    num_posts = df['enc_post_id'].nunique()
    k = 100
    l2_lambda = 0.001

    u_input = Input((1,), name='user_input')
    u = Embedding(num_users, k, name='user_emb')(u_input)
    u = Flatten(name='user_flat')(u)
    u = Dense(48, activation='relu', name='user_dense')(u)

    p_input = Input((1,), name='post_input')
    p = Embedding(num_posts, k, name='post_emb')(p_input)
    p = Flatten(name='post_flat')(p)
    p = Dense(8, activation='relu', name='post_dense')(p)

    x = concatenate([u, p], name='concat')
    # x = Dropout(0.1, name='drop1')(x)
    x = Dense(16, activation='relu', name='dense1')(x)
    x = Dense(4, activation='relu', name='dense2')(x)

    u_bias = Embedding(num_users, 1, embeddings_regularizer=l2(l2_lambda), name='user_bias_emb')(u_input)
    u_bias = Flatten(name='user_bias_flat')(u_bias)

    p_bias = Embedding(num_posts, 1, embeddings_regularizer=l2(l2_lambda), name='post_bias_emb')(p_input)
    p_bias = Flatten(name='post_bias_flat')(p_bias)

    o = concatenate([x, u_bias, p_bias], name='combined_features')
    o = Dense(16, activation='relu', name='combined_dense1')(o)
    o = Dense(4, activation='relu', name='combined_dense2')(o)
    o = Dense(1, activation='linear' , name='output')(o)

    model = Model(inputs=[u_input, p_input], outputs=o)

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.001), loss='mean_squared_error')

    history = model.fit(
        x=[X['enc_user_id'], X['enc_post_id']],
        y=y, epochs=200, verbose=0
    )

    tf.keras.models.save_model(model, 'models/posts/recommendations.h5')

    #* Saving Embeddings
    user_embeddings = {}

    for enc_user_id in df['enc_user_id'].unique():
        user_embedding = model.get_layer('user_emb')(np.array([enc_user_id]))
        user_embedding = tf.keras.backend.flatten(user_embedding)
        user_embedding = tf.expand_dims(user_embedding, axis=0)
        user_dense = model.get_layer('user_dense')(user_embedding)
        
        user_embeddings[user_le.inverse_transform([enc_user_id])[0]] = user_dense.numpy().tolist()

    post_embeddings = {}

    for enc_post_id in df['enc_post_id'].unique():
        post_embedding = model.get_layer('post_emb')(np.array([enc_post_id]))
        post_embedding = tf.keras.backend.flatten(post_embedding)
        post_embedding = tf.expand_dims(post_embedding, axis=0)
        post_dense = model.get_layer('post_dense')(post_embedding)
        
        post_embeddings[post_le.inverse_transform([enc_post_id])[0]] = post_dense.numpy().tolist()

    user_bias_embeddings = {}

    for enc_user_id in df['enc_user_id'].unique():
        user_bias_embedding = model.get_layer('user_bias_emb')(np.array([enc_user_id]))
        user_bias_embedding = tf.keras.backend.flatten(user_bias_embedding)

        user_bias_embeddings[user_le.inverse_transform([enc_user_id])[0]] = user_bias_embedding.numpy().tolist()

    post_bias_embeddings = {}

    for enc_post_id in df['enc_post_id'].unique():
        post_bias_embedding = model.get_layer('post_bias_emb')(np.array([enc_post_id]))
        post_bias_embedding = tf.keras.backend.flatten(post_bias_embedding)
        
        post_bias_embeddings[post_le.inverse_transform([enc_post_id])[0]] = post_bias_embedding.numpy().tolist()

    with open('models/posts/user_embeddings.json', 'w') as f:
        json.dump(user_embeddings, f)

    with open('models/posts/post_embeddings.json', 'w') as f:
        json.dump(post_embeddings, f)

    with open('models/posts/user_bias_embeddings.json', 'w') as f:
        json.dump(user_bias_embeddings, f)

    with open('models/posts/post_bias_embeddings.json', 'w') as f:
        json.dump(post_bias_embeddings, f)

    logger("info",f"Training Successful", "Successfully Trained Recommended Posts", "scripts/posts/recommendation.py")
except Exception as e :
    logger("error",f"Training Failed", str(e), "scripts/posts/recommendation.py")
