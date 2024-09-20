from celery import Celery


def setup_celery(app_name=__name__):
    celery = Celery(
        app_name,
        broker="redis://redis:6379/0",
        backend="redis://redis:6379/0",
        include=["tasks.code_review"],  # This points to the file with your Celery task
    )

    # Load any additional config
    celery.conf.update(
        result_expires=3600,  # Example of adding a config
    )
    return celery


app = setup_celery(app_name="interact-tasks")
