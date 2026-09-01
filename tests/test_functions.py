import numpy as np
from functions import calculate_performance


def fake_sentiment_task(text):
    mapping = {
        "good": [{"label": "positive"}],
        "bad": [{"label": "negative"}],
        "okay": [{"label": "neutral"}],
    }
    return mapping[text]


def test_calculate_performance_basic():
    X = np.array(["good", "bad", "okay"])
    y = np.array(["positive", "negative", "neutral"])

    sentiment, model_metrics = calculate_performance(
        X=X,
        y=y,
        sentiment_task=fake_sentiment_task,
        review_for_week=3,
        current_week=1
    )

    assert sentiment == {
        "negative": 1,
        "neutral": 1,
        "positive": 1,
    }

    assert "accuracy_score" in model_metrics
    assert "classification_report" in model_metrics
    assert "confusion_matrix" in model_metrics

    assert model_metrics["accuracy_score"] == 1.0
    assert model_metrics["confusion_matrix"].shape == (3, 3)
