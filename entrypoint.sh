#!/bin/bash 

# Run cron in the background
cron &

# Start FastAPI using uvicorn
uvicorn api:app --host 0.0.0.0 --port 3030

# Keep the script running to prevent the container from exiting
tail -f /dev/null
