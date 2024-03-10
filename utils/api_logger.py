import json
import requests
import jwt
from datetime import datetime, timedelta
import os

LOGGER_URL = os.getenv("LOGGER_URL")
LOGGER_SECRET = os.getenv("LOGGER_SECRET")
LOGGER_TOKEN = os.getenv("LOGGER_TOKEN")

class LogEntrySchema:
    def __init__(self, level, title, description, path, timestamp):
        self.level = level
        self.title = title
        self.description = description
        self.path = path
        self.timestamp = timestamp

def create_admin_jwt():
    token_claim = jwt.encode({
        'sub': 'backend',
        'crt': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(seconds=15.0)
    }, LOGGER_SECRET, algorithm='HS256')

    return token_claim

def log_to_admin_logger(record):
    #TODO add path and title to logs
    log_entry = LogEntrySchema(
        level=record.levelname,
        title=record.msg,
        description=record.msg,
        path='path/to/log',
        timestamp=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    )

    json_data = json.dumps(log_entry.__dict__)

    jwt_token = create_admin_jwt()

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + jwt_token,
        'API-TOKEN': LOGGER_TOKEN
    }

    response = requests.post(LOGGER_URL, headers=headers, data=json_data)

    if response.status_code != 200:
        print('Error:', response.text)
