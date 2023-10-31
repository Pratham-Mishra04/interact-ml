import pandas as pd
import pickle
from keras.models import load_model
from tensorflow import newaxis

#* Similar Projects
def find_similar(project_id):
    try:
        df = pd.read_csv('../data/projects.csv')
        with open('../models/projects/similarities.pickle', 'rb') as f:
            similarities=pickle.load(f)

        movie_index = df[df['id'].str.lower()==project_id.lower()].index[0]
        distances = similarities[movie_index]
        movie_objs = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

        return [df.iloc[i[0]].id for i in movie_objs]
    except:
        return []
    
def similar(body):
    recommendations = find_similar(body.id)
    return {
        'recommendations':recommendations
    }

#* Project Recommendations
model = load_model('../../models/projects/recommendations.h5')

with open('../../models/projects/user_embeddings.json', 'rb') as f:
    user_embeddings = pickle.load(f)

with open('../../models/projects/movie_embeddings.json', 'rb') as f:
    project_embeddings = pickle.load(f)

with open('../../models/projects/user_bias_embeddings.json', 'rb') as f:
    user_bias_embeddings = pickle.load(f)

with open('../../models/projects/movie_bias_embeddings.json', 'rb') as f:
    project_bias_embeddings = pickle.load(f)

def predict_score(user_id, project_id):
    user_embedding = user_embeddings[user_id]
    project_embedding = project_embeddings[project_id]

    # Passing user embedding and movie embedding through the concat layer
    concatenated_embeddings = model.get_layer('concat')([user_embedding, project_embedding])

    # Passing the concatenated embeddings through the dense layers
    x = model.get_layer('dense1')(concatenated_embeddings)
    x = model.get_layer('dense2')(x)

    user_bias_embedding = user_bias_embeddings[user_id]
    project_bias_embedding = project_bias_embeddings[project_id]

    user_bias_embedding = user_bias_embedding[:, newaxis]
    project_bias_embedding = project_bias_embedding[:, newaxis]

    # Combine embeddings, biases, and pass through the output layer
    input_tensors = [x, user_bias_embedding, project_bias_embedding]
    concatenated_features = model.get_layer('combined_features')(input_tensors)
    
    x = model.get_layer('combined_dense1')(concatenated_features)
    x = model.get_layer('combined_dense2')(x)
    
    x = model.get_layer('output')(x)

    predicted_rating = x[0][0]
    return predicted_rating.numpy()

def recommend(body):
    df = pd.read_csv('../data/project_scores.csv')
    user_ratings = df[df['user_id'] == body.id]
    user_ratings = df[df['project_id'] != 1]
    recommendation = df[~df['project_id'].isin(user_ratings['project_id'])][['project_id']].drop_duplicates()
    recommendation['score_predict'] = recommendation.apply(lambda x: predict_score(body.id, x['project_id']), axis=1)
    
    final_rec = recommendation.sort_values(by='score_predict', ascending=False)
    return final_rec['score_predict'].values