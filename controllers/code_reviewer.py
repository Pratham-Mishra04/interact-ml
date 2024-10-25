from tasks.code_review import process_code_review_async


def review_code(body):
    if body is None or body.repo_links is None or body.callback_url is None:
        return {
            "message": "Repository links and callback URL are required",
            "error": "Bad Request",
        }

    repos = body.repo_links
    callback_url = body.callback_url
    cloneRepoPath = "code_review/cloned_repos"

    # Trigger the Celery background task
    process_code_review_async.delay(repos, cloneRepoPath, callback_url)

    # Return an immediate response to the client
    return {
        "status": "processing",
        "message": "The code review process has started and will notify the result to the callback URL.",
    }
