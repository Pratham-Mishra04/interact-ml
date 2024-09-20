from code_review.src.chunker2.chunk_extractor import ChunkExtractor2
from code_review.src.code_analyser.code_analyser import CodeAnalyser
from code_review.src.fetcher.git_handler import GitHandler
from code_review.src.fetcher.repository_manager import RepositoryManager
import requests
import concurrent.futures
from .config import app


@app.task
def process_code_review_async(repos, cloneRepoPath, callback_url):
    """
    Celery task to process the code review asynchronously and send results to a callback URL.
    This task retries up to 5 times in case of errors, with a 10-second delay between retries.
    """
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
    """
    Function to send results or errors back to the provided callback URL.
    Retries can also be added here, or you can let Celery handle retries by failing the task.
    """
    try:
        response = requests.post(callback_url, json=result)
        if response.status_code == 200:
            print(f"Callback successfully sent to {callback_url}")
        else:
            print(
                f"Failed to send callback to {callback_url}, status code: {response.status_code}"
            )
    except Exception as e:
        print(f"Error sending callback to {callback_url}: {str(e)}")
