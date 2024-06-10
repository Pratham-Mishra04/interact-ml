import json
import requests
import jwt
from datetime import datetime, timedelta
import os
import logging
import sys

LOGGER_URL = os.getenv("LOGGER_URL")
LOGGER_SECRET = os.getenv("LOGGER_SECRET")
LOGGER_TOKEN = os.getenv("LOGGER_TOKEN")

def create_logger(name, filename, level, format):
    logger = logging.Logger(name, level)
    logger.setLevel(level)
    formatter = logging.Formatter(format)

    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

info_logger = create_logger('info_logger',filename="../logs/info.log", level=logging.INFO, format='%(asctime)s %(message)s' )
error_logger = create_logger('error_logger',filename="../logs/error.log", level=logging.INFO, format='%(asctime)s %(message)s' )

class LogEntrySchema:
    def __init__(self, level, title, description, path, timestamp):
        self.level = level
        self.title = title
        self.description = description
        self.path = path
        self.timestamp = timestamp

def create_admin_jwt():
    token_claim = jwt.encode({
        'sub': 'ml',
        'crt': datetime.utcnow().timestamp(),
        'exp': (datetime.utcnow() + timedelta(seconds=15.0)).timestamp()
    }, LOGGER_SECRET, algorithm='HS256')

    return token_claim

def log_to_admin_logger(record):
    try:
        log_entry = LogEntrySchema(
            level=record['level'],
            title=record['title'],
            description=record['description'],
            path=record['path'],
            timestamp=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        )

        json_data = json.dumps(log_entry.__dict__)

        jwt_token = create_admin_jwt()

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + jwt_token,
            'api-token': LOGGER_TOKEN
        }

        response = requests.post(LOGGER_URL, headers=headers, data=json_data)

        if response.status_code != 200:
            pass
            # error_logger.error(f"Title: Error Posting to Admin Logger, Description: {response.text}, Path: utils/api_logger.py")

    except Exception as e:
        pass
        # error_logger.error(f"Title: Error Posting to Admin Logger, Description: {str(e)}, Path: utils/api_logger.py")

if __name__ =="__main__":
    record = {
        'level': sys.argv[1] if len(sys.argv) > 1 else '',
        'title': sys.argv[2] if len(sys.argv) > 2 else '',
        'description': sys.argv[3] if len(sys.argv) > 3 else '',
        'path': sys.argv[4] if len(sys.argv) > 4 else '',
    }

    log_to_admin_logger(record)
    
    if record['level'] =='error':
        error_logger.error(f"Title: {record['title']}, Description: {record['description']}, Path: {record['path']}")
    else:
        info_logger.info(f"Title: {record['title']}, Description: {record['description']}, Path: {record['path']}")
