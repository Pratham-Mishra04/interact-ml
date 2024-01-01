#!/bin/bash

# Run cron in the foreground
cron -f &

# Start FastAPI using uvicorn in the background
uvicorn api:app --host 0.0.0.0 --port 3030

# Keep the script running to prevent the container from exiting
tail -f /dev/null
