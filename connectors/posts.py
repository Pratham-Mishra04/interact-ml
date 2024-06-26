import psycopg2
import os
from dotenv import load_dotenv
import csv
import subprocess

load_dotenv()

def logger(level , title, description, path):
    subprocess.run(['python3', 'api_logger.py', level, title, description, path], cwd='utils')

try:
    conn = psycopg2.connect(database=os.getenv("DB_NAME"),
                            user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASS"),
                            host=os.getenv("DB_HOST"),
                            port=os.getenv("DB_PORT"))

    cursor = conn.cursor()

    user_post2rating = {}

    # Likes
    cursor.execute('SELECT user_id, post_id from likes WHERE post_id IS NOT NULL')
    likes=cursor.fetchall()

    for i in likes:
        userID, postID = i
        user_post2rating[userID+' '+postID]=1

    # Messages
    cursor.execute('SELECT user_id, post_id from messages WHERE post_id IS NOT NULL')
    messages=cursor.fetchall()

    for i in messages:
        userID, postID = i
        user_post2rating[userID+' '+postID]=2

    # Bookmarks
    cursor.execute('''
                SELECT pb.user_id, pbi.post_id
                FROM post_bookmarks pb
                JOIN post_bookmark_items pbi
                ON pb.id = pbi.post_bookmark_id;
                ''')
    bookmark_items=cursor.fetchall()

    for i in bookmark_items:
        userID, postID = i
        user_post2rating[userID+' '+postID]=3

    # # Dislikes
    # cursor.execute('SELECT user_id, post_id from dislikes WHERE post_id IS NOT NULL')
    # reports=cursor.fetchall()

    # for i in reports:
    #     userID, postID = i
    #     user_post2rating[userID+' '+postID]=-1

    # Reports
    cursor.execute('SELECT reporter_id, post_id from reports WHERE post_id IS NOT NULL')
    reports=cursor.fetchall()

    for i in reports:
        userID, postID = i
        user_post2rating[userID+' '+postID]=-3


    writer = csv.writer(open("data/post_scores.csv", 'w'))
    writer.writerow(["user_id", "post_id", "score"])

    for i in user_post2rating:
        user_id = i.split(' ')[0]
        post_id = i.split(' ')[1]
        score = user_post2rating[i]
        writer.writerow([user_id, post_id, score])

    logger("info",f"Training Successful", "Successfully fetched Posts", "connectors/posts.py")
except Exception as e:
    logger("error",f"Training Failed", str(e), "connectors/posts.py")
