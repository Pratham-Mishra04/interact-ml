from code_review.src.chunker2.chunk_extractor import ChunkExtractor2
from code_review.src.code_analyser.code_analyser import CodeAnalyser
from code_review.src.fetcher.git_handler import GitHandler
from code_review.src.fetcher.repository_manager import RepositoryManager
import requests
import concurrent.futures
from .config import app
import os
from dotenv import load_dotenv
import subprocess

load_dotenv()


def logger(level, title, description, path):
    subprocess.run(
        ["python3", "api_logger.py", level, title, description, path], cwd="utils"
    )


@app.task
def process_code_review_async(repos, cloneRepoPath, callback_url):
    git_handler = GitHandler()
    repo_manager = RepositoryManager(git_handler)
    chunk_extractor = ChunkExtractor2()
    code_analyser = CodeAnalyser()

    def process_single_repo(repo):
        try:
            # Clone repository
            repo_manager.clone_repository(repo, cloneRepoPath)
            return repo
        except Exception as e:
            print(f"Error cloning repo {repo}: {e}")
            return None

    try:
        # Parallelize repo cloning
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_single_repo, repo) for repo in repos]
            results = concurrent.futures.wait(futures)
            successful_repos = [
                future.result() for future in futures if future.result() is not None
            ]

        if not successful_repos:
            raise Exception("No repos were successfully cloned for processing.")

        # Extract chunks and analyze repos only for successfully cloned repos
        chunk_extractor.processRepos(cloneRepoPath)
        scores = code_analyser.processAllRepos(cloneRepoPath)

        # Clean up cloned repos
        repo_manager.complete_cleanup()

        # Send the result to the callback URL
        result = {"repo_links": successful_repos, "scores": scores}
        send_callback(callback_url, result)

    except Exception as e:
        # Retry logic if something fails, handled by Celery's autoretry mechanism
        error_result = {"message": "Error fetching code review", "error": str(e)}
        send_callback(callback_url, error_result)


def send_callback(callback_url, result):
    try:
        api_token = os.getenv("ML_API_TOKEN")

        if not api_token:
            logger(
                "error",
                "Missing ML-API-TOKEN",
                "ML-API-TOKEN environment variable is not set. Cannot send callback without the token.",
                "code-review-callback",
            )
            return

        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

        response = requests.post(callback_url, json=result, headers=headers)
        if response.status_code == 200:
            logger(
                "info",
                "Callback Success",
                f"Callback successfully sent to {callback_url}. Response: {response.status_code} - {response.reason}",
                "code-review-callback",
            )
        else:
            logger(
                "error",
                "Callback Failed",
                f"Failed to send callback to {callback_url}. Received status code: {response.status_code}. Response message: {response.text}",
                "code-review-callback",
            )
    except requests.exceptions.RequestException as e:
        # Handles all request-related errors (timeout, connection errors, etc.)
        logger(
            "error",
            "Callback Request Exception",
            f"RequestException occurred while sending callback to {callback_url}. Error details: {str(e)}",
            "code-review-callback",
        )
    except Exception as e:
        # Catches any other unexpected exceptions
        logger(
            "error",
            "Unexpected Exception During Callback",
            f"An unexpected error occurred while sending callback to {callback_url}. Error details: {str(e)}",
            "code-review-callback",
        )
