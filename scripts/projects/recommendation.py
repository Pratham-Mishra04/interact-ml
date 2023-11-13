import pandas as pd
import numpy as np
import json
import logging

logging.basicConfig(filename="logs/training.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a')
training_logger = logging.getLogger('training_logger')

training_logger.info("Projects-Recommendation: Training Started")

try:
    #* Model Building
    df = pd.read_csv('data/project_scores.csv')

    from sklearn.preprocessing import LabelEncoder

    user_le = LabelEncoder()
    df['enc_user_id'] = user_le.fit_transform(df['user_id'])

    project_le = LabelEncoder()
    df['enc_project_id'] = project_le.fit_transform(df['project_id'])

    X = df[['enc_user_id', 'enc_project_id']]
    y = df['score']

    import tensorflow as tf
    from keras.models import Model
    from keras.layers import Input,Dense,Embedding,Flatten,Input,concatenate
    from keras.regularizers import l2

    num_users = df['enc_user_id'].nunique()
    num_projects = df['enc_project_id'].nunique()
    k = 100
    l2_lambda = 0.001

    u_input = Input((1,), name='user_input')
    u = Embedding(num_users, k, name='user_emb')(u_input)
    u = Flatten(name='user_flat')(u)
    u = Dense(48, activation='relu', name='user_dense')(u)

    p_input = Input((1,), name='project_input')
    p = Embedding(num_projects, k, name='project_emb')(p_input)
    p = Flatten(name='project_flat')(p)
    p = Dense(8, activation='relu', name='project_dense')(p)

    x = concatenate([u, p], name='concat')
    # x = Dropout(0.1, name='drop1')(x)
    x = Dense(16, activation='relu', name='dense1')(x)
    x = Dense(4, activation='relu', name='dense2')(x)

    u_bias = Embedding(num_users, 1, embeddings_regularizer=l2(l2_lambda), name='user_bias_emb')(u_input)
    u_bias = Flatten(name='user_bias_flat')(u_bias)

    p_bias = Embedding(num_projects, 1, embeddings_regularizer=l2(l2_lambda), name='project_bias_emb')(p_input)
    p_bias = Flatten(name='project_bias_flat')(p_bias)

    o = concatenate([x, u_bias, p_bias], name='combined_features')
    o = Dense(16, activation='relu', name='combined_dense1')(o)
    o = Dense(4, activation='relu', name='combined_dense2')(o)
    o = Dense(1, activation='linear' , name='output')(o)

    model = Model(inputs=[u_input, p_input], outputs=o)

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.001), loss='mean_squared_error')

    history = model.fit(
        x=[X['enc_user_id'], X['enc_project_id']],
        y=y, epochs=200,verbose=0
    )

    tf.keras.models.save_model(model, 'models/projects/recommendations.h5')

    #* Saving Embeddings
    user_embeddings = {}

    for enc_user_id in df['enc_user_id'].unique():
        user_embedding = model.get_layer('user_emb')(np.array([enc_user_id]))
        user_embedding = tf.keras.backend.flatten(user_embedding)
        user_embedding = tf.expand_dims(user_embedding, axis=0)
        user_dense = model.get_layer('user_dense')(user_embedding)
        
        user_embeddings[user_le.inverse_transform([enc_user_id])[0]] = user_dense.numpy().tolist()

    project_embeddings = {}

    for enc_project_id in df['enc_project_id'].unique():
        project_embedding = model.get_layer('project_emb')(np.array([enc_project_id]))
        project_embedding = tf.keras.backend.flatten(project_embedding)
        project_embedding = tf.expand_dims(project_embedding, axis=0)
        project_dense = model.get_layer('project_dense')(project_embedding)
        
        project_embeddings[project_le.inverse_transform([enc_project_id])[0]] = project_dense.numpy().tolist()

    user_bias_embeddings = {}

    for enc_user_id in df['enc_user_id'].unique():
        user_bias_embedding = model.get_layer('user_bias_emb')(np.array([enc_user_id]))
        user_bias_embedding = tf.keras.backend.flatten(user_bias_embedding)

        user_bias_embeddings[user_le.inverse_transform([enc_user_id])[0]] = user_bias_embedding.numpy().tolist()

    project_bias_embeddings = {}

    for enc_project_id in df['enc_project_id'].unique():
        project_bias_embedding = model.get_layer('project_bias_emb')(np.array([enc_project_id]))
        project_bias_embedding = tf.keras.backend.flatten(project_bias_embedding)
        
        project_bias_embeddings[project_le.inverse_transform([enc_project_id])[0]] = project_bias_embedding.numpy().tolist()

    with open('models/projects/user_embeddings.json', 'w') as f:
        json.dump(user_embeddings, f)

    with open('models/projects/project_embeddings.json', 'w') as f:
        json.dump(project_embeddings, f)

    with open('models/projects/user_bias_embeddings.json', 'w') as f:
        json.dump(user_bias_embeddings, f)

    with open('models/projects/project_bias_embeddings.json', 'w') as f:
        json.dump(project_bias_embeddings, f)
        
    training_logger.info("Projects-Recommendation: Training Completed")
except Exception as e :
    training_logger.error(f"Projects-Recommendation: An error occurred- {str(e)}")