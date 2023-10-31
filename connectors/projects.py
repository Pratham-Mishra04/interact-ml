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

cursor.execute('SELECT * from projects')

projects=cursor.fetchall()

writer = csv.writer(open("projects.csv", 'w'))
writer.writerow(["id", "title", "tagline", "description", "userID", "tags", "category"])

for p in projects:
    id, title, tagline, description, userID, tags, category = (p[0], p[1], p[2], p[5], p[7], p[9], p[13])
    writer.writerow([id, title, tagline, description, userID, tags, category])