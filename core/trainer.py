# core/trainer.py

import time
from sklearn.pipeline import Pipeline


def train_model(preprocessor, model, X_train, y_train):

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    start = time.time()
    full_pipeline.fit(X_train, y_train)
    end = time.time()

    training_time = end - start

    return full_pipeline, training_time