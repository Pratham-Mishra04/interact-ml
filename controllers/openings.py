import pandas as pd
import pickle

#* Similar Openings
def find_similar(opening_id):
    try:
        df = pd.read_csv('data/openings.csv')
        with open('models/openings/similarities.pickle', 'rb') as f:
            similarities=pickle.load(f)

        opening_index = df[df['id'].str.lower()==opening_id.lower()].index[0]
        distances = similarities[opening_index]
        opening_objs = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

        return [df.iloc[i[0]].id for i in opening_objs]
    except:
        return []
    
def similar(body):
    recommendations = find_similar(body.id)
    return {
        'recommendations':recommendations
    }