import psycopg2
import os
from dotenv import load_dotenv
import csv
import logging

load_dotenv()

logging.basicConfig(filename="logs/training.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a')
training_logger = logging.getLogger('training_logger')

training_logger.info("Openings-Connector: Data Fetching Started")

try:
    try:
        conn = psycopg2.connect(database=os.getenv("DB_NAME"),
                                user=os.getenv("DB_USER"),
                                password=os.getenv("DB_PASS"),
                                host=os.getenv("DB_HOST"),
                                port=os.getenv("DB_PORT"))
        print("Connected to DB")
    except Exception as e:
        print("Not Connected to DB", e)

    cursor = conn.cursor()

    cursor.execute('SELECT id, project_id, title, description, tags from openings')

    openings=cursor.fetchall()

    writer = csv.writer(open("data/openings.csv", 'w'))
    writer.writerow(["id", "project_id", "title", "description", "tags"])

    for o in openings:
        id, project_id, title, description, tags = o
        writer.writerow([id, project_id, title, description, tags])

    user_opening2rating = {}

    # Views
    cursor.execute('SELECT user_id, opening_id from last_viewed_openings')
    views=cursor.fetchall()

    for i in views:
        userID, openingID = i
        user_opening2rating[userID+' '+openingID]=1

    # Messages
    cursor.execute('SELECT user_id, opening_id from messages WHERE opening_id IS NOT NULL')
    messages=cursor.fetchall()

    for i in messages:
        userID, openingID = i
        user_opening2rating[userID+' '+openingID]=2

    # Bookmarks
    cursor.execute('''
                SELECT ob.user_id, obi.opening_id
                FROM opening_bookmarks ob
                JOIN opening_bookmark_items obi
                ON ob.id = obi.opening_bookmark_id;
                ''')
    bookmark_items=cursor.fetchall()

    for i in bookmark_items:
        userID, openingID = i
        user_opening2rating[userID+' '+openingID]=3

    # Applications
    cursor.execute('SELECT user_id, opening_id from applications')
    bookmark_items=cursor.fetchall()

    for i in bookmark_items:
        userID, openingID = i
        user_opening2rating[userID+' '+openingID]=4

    # Reports
    cursor.execute('SELECT reporter_id, opening_id from reports WHERE opening_id IS NOT NULL')
    reports=cursor.fetchall()

    for i in reports:
        userID, openingID = i
        user_opening2rating[userID+' '+openingID]=-1


    writer = csv.writer(open("data/opening_scores.csv", 'w'))
    writer.writerow(["user_id", "opening_id", "score"])

    for i in user_opening2rating:
        user_id = i.split(' ')[0]
        project_id = i.split(' ')[1]
        score = user_opening2rating[i]
        writer.writerow([user_id, project_id, score])

    training_logger.info("Openings-Connector: Data Fetching Finished")
except Exception as e:
    training_logger.error(f"Openings-Connector: An error occurred- {str(e)}")