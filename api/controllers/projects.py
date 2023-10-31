import pandas as pd
import pickle

def recommend(project_id):
    try:
        df = pd.read_csv('../data/projects.csv')
        with open('../models/projects/similarities.pickle', 'rb') as f:
            similarities=pickle.load(f)

        movie_index = df[df['id'].str.lower()==project_id.lower()].index[0]
        distances = similarities[movie_index]
        movie_objs = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

        return [(df.iloc[i[0]].id, df.iloc[i[0]].title) for i in movie_objs]
    except:
        return []
    
def similar_projects(body):
    recommendations = recommend(body.id)
    return {
        'recommendations':recommendations
    }