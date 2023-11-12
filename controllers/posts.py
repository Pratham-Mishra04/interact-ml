import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow import newaxis
import json

#* Post Recommendations
model = load_model('models/posts/recommendations.h5')

with open('models/posts/user_embeddings.json') as f:
    user_embeddings = json.load(f)
with open('models/posts/post_embeddings.json') as f:
    post_embeddings = json.load(f)
with open('models/posts/user_bias_embeddings.json') as f:
    user_bias_embeddings = json.load(f)
with open('models/posts/post_bias_embeddings.json') as f:
    post_bias_embeddings = json.load(f)

def predict_score(user_id, post_id):
    user_embedding = np.array(user_embeddings[user_id])
    post_embedding = np.array(post_embeddings[post_id])

    # Passing user embedding and movie embedding through the concat layer
    concatenated_embeddings = model.get_layer('concat')([user_embedding, post_embedding])

    # Passing the concatenated embeddings through the dense layers
    x = model.get_layer('dense1')(concatenated_embeddings)
    x = model.get_layer('dense2')(x)

    user_bias_embedding = np.array(user_bias_embeddings[user_id])
    post_bias_embedding = np.array(post_bias_embeddings[post_id])

    user_bias_embedding = user_bias_embedding[:, newaxis]
    post_bias_embedding = post_bias_embedding[:, newaxis]

    # Combine embeddings, biases, and pass through the output layer
    input_tensors = [x, user_bias_embedding, post_bias_embedding]
    concatenated_features = model.get_layer('combined_features')(input_tensors)
    
    x = model.get_layer('combined_dense1')(concatenated_features)
    x = model.get_layer('combined_dense2')(x)
    
    x = model.get_layer('output')(x)

    predicted_rating = x[0][0]
    return predicted_rating.numpy()

def recommend(body):
    df = pd.read_csv('data/post_scores.csv')
    user_ratings = df[df['user_id'] == body.id]
    recommendation = df[~df['post_id'].isin(user_ratings['post_id'])][['post_id']].drop_duplicates()
    recommendation['score_predict'] = recommendation.apply(lambda x: predict_score(body.id, x['post_id']), axis=1)
    
    final_rec = recommendation.sort_values(by='score_predict', ascending=False)
    return {
        'recommendations':final_rec['post_id'].values.tolist()
    }