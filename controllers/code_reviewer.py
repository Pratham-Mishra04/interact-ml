from tasks.code_review import process_code_review_async
import os


def review_code(body, request):
    if body is None or body.repo_links is None or body.callback_url is None:
        return {
            "message": "Repository links and callback URL are required",
            "error": "Bad Request",
        }

    BACKEND_TOKEN = os.getenv("BACKEND_TOKEN")

    token = request.headers.get("api-token")

    if not token or token != BACKEND_TOKEN:
        return {
            "message": "Cannot access this route.",
            "error": "Bad Request",
        }

    repos = body.repo_links
    callback_url = body.callback_url
    cloneRepoPath = "code_review/cloned_repos"

    # Trigger the Celery background task
    process_code_review_async.delay(repos, cloneRepoPath, callback_url)

    return {
        "status": "processing",
        "message": "The code review process has started and will notify the result to the callback URL.",
    }
