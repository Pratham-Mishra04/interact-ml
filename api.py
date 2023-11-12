from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import controllers.projects as project_controllers
import controllers.posts as post_controllers
import controllers.openings as opening_controllers
import os

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReqBody(BaseModel):
    id:str

@app.get("/ping/{input_text}")
def ping(input_text: str):
    return {"pongs":input_text}

@app.post('/projects/similar')
def similar_projects(body:ReqBody):
    return project_controllers.similar(body)

@app.post('/openings/similar')
def similar_openings(body:ReqBody):
    return opening_controllers.similar(body)

@app.post('/projects/recommend')
def recommend_projects(body:ReqBody):
    return project_controllers.recommend(body)

@app.post('/posts/recommend')
def recommend_posts(body:ReqBody):
    return post_controllers.recommend(body)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT"), reload=True)