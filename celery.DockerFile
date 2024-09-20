# Use an official Python runtime as a parent image
FROM python:3.10

# Create a non-root user
RUN useradd -ms /bin/bash celeryuser

# Set the working directory to /app
WORKDIR /app

# Copy the main requirements file and install dependencies
COPY celery.requirements.txt /app/
RUN pip install -r celery.requirements.txt

# Copy the code_review requirements file and install its dependencies
COPY code_review/requirements.txt /app/code_review/requirements.txt
RUN pip install -r /app/code_review/requirements.txt

# Copy only the necessary source files
COPY code_review /app/code_review
COPY tasks /app/tasks
COPY utils /app/utils
COPY .env /app/.env

# Change ownership of the /app directory to the non-root user
RUN chown -R celeryuser:celeryuser /app

# Switch to the non-root user
USER celeryuser

# Set the entrypoint for the Celery worker
CMD ["celery", "-A", "tasks.config.app", "worker", "--loglevel=info"]
