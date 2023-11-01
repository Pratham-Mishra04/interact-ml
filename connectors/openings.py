import psycopg2
import os
from dotenv import load_dotenv
import csv

load_dotenv()

try:
    conn = psycopg2.connect(database=os.getenv("DB_NAME"),
                            user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASS"),
                            host=os.getenv("DB_HOST"),
                            port=os.getenv("DB_PORT"))
    print("Database connected successfully")
except:
    print("Database not connected successfully")
    exit()

cursor = conn.cursor()

cursor.execute('SELECT id, project_id, title, description, tags from openings')

openings=cursor.fetchall()

writer = csv.writer(open("data/openings.csv", 'w'))
writer.writerow(["id", "project_id", "title", "description", "tags"])

for o in openings:
    id, project_id, title, description, tags = o
    writer.writerow([id, project_id, title, description, tags])