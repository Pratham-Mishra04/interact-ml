import psycopg2
import os
from dotenv import load_dotenv
import csv
import logging

load_dotenv()

logging.basicConfig(filename="logs/training.log", level=logging.INFO, format='%(asctime)s %(message)s', filemode='a')
training_logger = logging.getLogger('training_logger')

training_logger.info("Projects-Connector: Data Fetching Started")

try:

    conn = psycopg2.connect(database=os.getenv("DB_NAME"),
                                user=os.getenv("DB_USER"),
                                password=os.getenv("DB_PASS"),
                                host=os.getenv("DB_HOST"),
                                port=os.getenv("DB_PORT"))

    cursor = conn.cursor()

    cursor.execute('SELECT id, title, tagline, description, user_id, tags, category from projects')

    projects=cursor.fetchall()

    writer = csv.writer(open("data/projects.csv", 'w'))
    writer.writerow(["id", "title", "tagline", "description", "userID", "tags", "category"])

    for p in projects:
        id, title, tagline, description, userID, tags, category = p
        writer.writerow([id, title, tagline, description, userID, tags, category])

    user_project2rating = {}

    # Views
    cursor.execute('SELECT user_id, project_id from last_vieweds')
    views=cursor.fetchall()

    for i in views:
        userID, projectID = i
        user_project2rating[userID+' '+projectID]=1

    # Likes
    cursor.execute('SELECT user_id, project_id from likes WHERE project_id IS NOT NULL')
    likes=cursor.fetchall()

    for i in likes:
        userID, projectID = i
        user_project2rating[userID+' '+projectID]=2

    # Messages
    cursor.execute('SELECT user_id, project_id from messages WHERE project_id IS NOT NULL')
    messages=cursor.fetchall()

    for i in messages:
        userID, projectID = i
        user_project2rating[userID+' '+projectID]=3

    # Bookmarks
    cursor.execute('''
                SELECT pb.user_id, pbi.project_id
                FROM project_bookmarks pb
                JOIN project_bookmark_items pbi
                ON pb.id = pbi.project_bookmark_id;
                ''')
    bookmark_items=cursor.fetchall()

    for i in bookmark_items:
        userID, projectID = i
        user_project2rating[userID+' '+projectID]=4

    # # Dislikes
    # cursor.execute('SELECT user_id, project_id from dislikes WHERE project_id IS NOT NULL')
    # reports=cursor.fetchall()

    # for i in reports:
    #     userID, projectID = i
    #     user_project2rating[userID+' '+projectID]=-1

    # Reports
    cursor.execute('SELECT reporter_id, project_id from reports WHERE project_id IS NOT NULL')
    reports=cursor.fetchall()

    for i in reports:
        userID, projectID = i
        user_project2rating[userID+' '+projectID]=-3


    writer = csv.writer(open("data/project_scores.csv", 'w'))
    writer.writerow(["user_id", "project_id", "score"])

    for i in user_project2rating:
        user_id = i.split(' ')[0]
        project_id = i.split(' ')[1]
        score = user_project2rating[i]
        writer.writerow([user_id, project_id, score])

    training_logger.info("Projects-Connector: Data Fetching Finished")
except Exception as e:
    training_logger.info(f"Projects-Connector: An error occurred- {str(e)}")