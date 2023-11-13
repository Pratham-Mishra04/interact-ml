#!/bin/bash

# Start FastAPI using uvicorn in the background
# uvicorn api:app --host 0.0.0.0 --port 3030 &

# Run cron in the foreground
cron

# Keep the script running to prevent the container from exiting
tail -f /dev/null
