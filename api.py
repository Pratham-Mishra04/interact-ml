from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import controllers.projects as project_controllers
import controllers.posts as post_controllers
import controllers.openings as opening_controllers
import controllers.image as img_controllers
import os 

from dotenv import load_dotenv

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

@app.get("/ping/{input_text}")
def ping(input_text: str):
    return {"pong":input_text}

@app.post('/projects/similar')
async def similar_projects(body:ReqBody):
    return project_controllers.similar(body)

@app.post('/openings/similar')
async def similar_openings(body:ReqBody):
    return opening_controllers.similar(body)

@app.post('/projects/recommend')
async def recommend_projects(body:ReqBody):
    return project_controllers.recommend(body)

@app.post('/posts/recommend')
async def recommend_posts(body:ReqBody):
    return post_controllers.recommend(body)

@app.post("/image_blur_hash")
async def get_blur_hash(image: UploadFile = File(...)):
    return img_controllers.generate_blurhash_data_url(image)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT"), reload=True)