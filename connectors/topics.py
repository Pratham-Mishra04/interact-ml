import psycopg2
import os
from dotenv import load_dotenv
import json
import subprocess

load_dotenv()

def logger(level , title, description, path):
    subprocess.run(['python3', 'api_logger.py', level, title, description, path], cwd='utils')

def handle_return(result, multiple=True):
    if result:
        if not multiple:
            if result[0][0] is not None:
                return result[0][0]
            return []
        return [x[0] for x in result]
    return []

def get_user_tags(conn, user_id):
    cursor = conn.cursor()
    query = "SELECT tags FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    return handle_return(cursor.fetchall(), False)

def get_user_searches(conn, user_id):
    cursor = conn.cursor()
    query = "SELECT query FROM search_queries WHERE user_id = %s"
    cursor.execute(query, (user_id,))
    return handle_return(cursor.fetchall(), False)

def get_user_following_tags(conn, user_id):
    cursor = conn.cursor()
    query = """
    SELECT u.tags
    FROM follow_followers ff
    JOIN users u ON ff.followed_id = u.id
    WHERE ff.follower_id = %s
    """
    cursor.execute(query, (user_id,))
    return handle_return(cursor.fetchall())

def get_opening_tags_for_user_applications(conn, user_id):
    cursor = conn.cursor()
    query = """
    SELECT o.tags
    FROM applications a
    JOIN openings o ON a.opening_id = o.id
    WHERE a.user_id = %s
    """
    cursor.execute(query, (user_id,))
    return handle_return(cursor.fetchall())

def get_organization_tags_for_user_memberships(conn, user_id):
    cursor = conn.cursor()
    query = """
    SELECT u.tags
    FROM organization_memberships om
    JOIN organizations o ON om.organization_id = o.id
    JOIN users u ON o.user_id = u.id
    WHERE om.user_id = %s
    """
    cursor.execute(query, (user_id,))
    return handle_return(cursor.fetchall())

def get_liked_posts_topics(conn, user_id):
    cursor = conn.cursor()
    query = """
    SELECT p.topics
    FROM posts p
    JOIN likes l ON p.id = l.post_id
    WHERE l.user_id = %s
    """
    cursor.execute(query, (user_id,))
    return handle_return(cursor.fetchall())

def get_liked_project_tags(conn, user_id):
    cursor = conn.cursor()
    query = """
    SELECT p.tags
    FROM projects p
    JOIN likes l ON p.id = l.project_id
    WHERE l.user_id = %s
    """
    cursor.execute(query, (user_id,))
    return handle_return(cursor.fetchall())

def get_liked_event_tags(conn, user_id):
    cursor = conn.cursor()
    query = """
    SELECT e.tags
    FROM events e
    JOIN likes l ON e.id = l.event_id
    WHERE l.user_id = %s
    """
    cursor.execute(query, (user_id,))
    return handle_return(cursor.fetchall())

def get_all_user_ids(conn):
    cursor = conn.cursor()
    query = """
    SELECT id
    FROM users
    """
    cursor.execute(query)
    return handle_return(cursor.fetchall())

try:
    conn = psycopg2.connect(database=os.getenv("DB_NAME"),
                            user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASS"),
                            host=os.getenv("DB_HOST"),
                            port=os.getenv("DB_PORT"))

    cursor = conn.cursor()

    user_ids = get_all_user_ids(conn)

    configs = []

    for user_id in user_ids:
        user_tags = get_user_tags(conn, user_id)
        user_searches = get_user_searches(conn ,user_id)
        user_following_tags = get_user_following_tags(conn, user_id)
        opening_tags_for_user_applications = get_opening_tags_for_user_applications(conn ,user_id)
        organization_tags_for_user_memberships = get_organization_tags_for_user_memberships(conn, user_id)
        liked_posts_topics = get_liked_posts_topics(conn, user_id)
        liked_project_tags = get_liked_project_tags(conn, user_id)
        liked_event_tags = get_liked_event_tags(conn, user_id)

        config = {
            'user':{
                'tags':user_tags,
                'weight':0.2,
                'type':'single'
            },
            'searches':{
                'tags':user_searches,
                'weight':0.1,
                'type':'single'
            },
            'followings':{
                'tags':user_following_tags,
                'weight':0.1,
                'type':'multiple'
            },
            'applied_openings':{
                'tags':opening_tags_for_user_applications,
                'weight':0.1,
                'type':'multiple'
            },
            'member_organisations':{
                'tags':organization_tags_for_user_memberships,
                'weight':0.1,
                'type':'multiple'
            },
            'liked_posts':{
                'tags':liked_posts_topics,
                'weight':0.2,
                'type':'multiple'
            },
            'liked_projects':{
                'tags':liked_project_tags,
                'weight':0.1,
                'type':'multiple'
            },
            'liked_events':{
                'tags':liked_event_tags,
                'weight':0.1,
                'type':'multiple'
            }
        }

        configs.append({user_id:config})

    with open('data/topics.json', 'w') as f:
        json.dump(configs, f)

    logger("info",f"Training Successful", "Successfully fetched Topics-Data", "connectors/topics.py")
except Exception as e:
    print(e)
    logger("error",f"Training Failed", str(e), "connectors/topics.py")
