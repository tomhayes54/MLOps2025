from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment(text):
    response = sentiment_pipeline(text)
    return response