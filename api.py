from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import controllers.projects as project_controllers
import controllers.posts as post_controllers
import controllers.openings as opening_controllers
import controllers.applications as application_controllers
import controllers.miscellaneous as miscellaneous_controllers
import os 
from typing import List
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import pipeline

load_dotenv()

app = FastAPI()

origins = [os.getenv("BACKEND_URL")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReqBody(BaseModel):
    id:str
    limit: int = 4
    page: int = 1 

class ContentBody(BaseModel):
    content:str

class ApplicationScoreBody(BaseModel):
    cover_letter: str
    profile_topics: List[str]
    resume_topics: List[str]
    is_resume_included: bool
    opening_description_topics: List[str]
    organization_values_topics: List[str]
    years_of_experience: int

miniLM_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
miniLM_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

roberta_sentiment_pipeline = pipeline("sentiment-analysis", model='cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest')
falconai_image_pipeline = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

# topic_model_dir = "../../models/posts/topics"

# topics_bert_tokenizer = AutoTokenizer.from_pretrained(topic_model_dir)
# topics_bert_model = AutoModel.from_pretrained(topic_model_dir)

# with open(f'{topic_model_dir}/mlb.pickle', 'rb') as f:
#         topics_mlb=pickle.load(f)

app.state.miniLM_tokenizer = miniLM_tokenizer
app.state.miniLM_model = miniLM_model

app.state.bert_tokenizer = bert_tokenizer
app.state.bert_model = bert_model

app.state.roberta_sentiment_pipeline = roberta_sentiment_pipeline
app.state.falconai_image_pipeline = falconai_image_pipeline

# app.state.topics_bert_tokenizer = topics_bert_tokenizer
# app.state.topics_bert_model = topics_bert_model
# app.state.topics_mlb = topics_mlb

@app.get("/ping/{input_text}")
def ping(input_text: str):
    return {"pong":input_text}

@app.post('/projects/similar')
async def similar_projects(body:ReqBody):
    return project_controllers.similar(body)

@app.post('/openings/similar')
async def similar_openings(body:ReqBody):
    return opening_controllers.similar(body)

@app.post('/openings/application_score')
async def application_score(body:ApplicationScoreBody, request: Request):
    return opening_controllers.get_application_score(body, request)

@app.post('/openings/application_score2')
async def application_score_test(body:ApplicationScoreBody, request: Request):
    return application_controllers.get_application_score(body, request)

@app.post('/projects/recommend')
async def recommend_projects(body:ReqBody):
    return project_controllers.recommend(body)

@app.post('/posts/recommend')
async def recommend_posts(body:ReqBody):
    return post_controllers.recommend(body)

# @app.post('/posts/topics')
# async def recommend_posts(body:ContentBody, request: Request):
#     return post_controllers.get_topics(body, request)

@app.post("/image_blur_hash")
async def get_blur_hash(image: UploadFile = File(...)):
    return miscellaneous_controllers.generate_blurhash_data_url(image)

@app.post('/toxicity')
async def check_toxicity(body:ContentBody, request: Request):
    return miscellaneous_controllers.check_toxicity(body, request)

@app.post('/image_profanity')
async def check_toxicity(image: UploadFile, request: Request):
    return miscellaneous_controllers.check_image_profanity(image, request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT"))