
from social_tools.sentiment_analysis import SentimentAnalysis
def test_nltk_sentiment():
    sa = SentimentAnalysis(tool='nltk')
    result = sa.analyze("This is awesome!")
    assert result[0]['compound'] > 0

def test_textblob_sentiment():
    sa = SentimentAnalysis(tool='textblob')
    result = sa.analyze("This is awesome!")
    assert result[0]['polarity'] > 0

def test_spacy_sentiment():
    sa = SentimentAnalysis(tool='spacy')
    result = sa.analyze("This is awesome!")
    assert result[0]['polarity'] > 0

def test_huggingface_sentiment():
    sa = SentimentAnalysis(tool='huggingface')
    result = sa.analyze("This is awesome!")
    assert result[0]['label'] == 'POSITIVE'
