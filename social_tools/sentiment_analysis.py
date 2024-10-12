import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import pipeline
from transformers.pipelines import PipelineException
import subprocess

# Download NLTK dependencies if not already downloaded
nltk.download('vader_lexicon', quiet=True)


class SentimentAnalysisNLTK:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment using NLTK's VADER model.

        Args:
            text (str): Input text to analyze.

        Returns:
            dict: Sentiment analysis result with polarity scores.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
        
        return self.analyzer.polarity_scores(text)


class SentimentAnalysisTextBlob:
    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment using TextBlob.

        Args:
            text (str): Input text to analyze.

        Returns:
            dict: Sentiment analysis result with polarity and subjectivity.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
        
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

class SentimentAnalysisSpaCy:
    def __init__(self):
        """
        Initialize SpaCy with the spacytextblob pipeline.
        If the SpaCy model is not available, download it automatically.
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If the model is not found, print a message and download the model
            print("SpaCy model 'en_core_web_sm' not found. Attempting to download it...")
            self.download_spacy_model()
            self.nlp = spacy.load("en_core_web_sm")

        # Ensure spacytextblob is added to the pipeline
        if "spacytextblob" not in self.nlp.pipe_names:
            self.nlp.add_pipe("spacytextblob")

    def download_spacy_model(self):
        """
        Download the SpaCy 'en_core_web_sm' model.
        """
        try:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            print("Model 'en_core_web_sm' downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download SpaCy model: {e}")
            raise OSError ("Spacy model was not donwlaoded. Check Spacy documentation for downloading models and try again")

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment using SpaCy with spacytextblob.

        Args:
            text (str): Input text to analyze.

        Returns:
            dict: Sentiment analysis result with polarity and subjectivity.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
        
        doc = self.nlp(text)
        return {
            'polarity': doc._.blob.polarity,
            'subjectivity': doc._.blob.subjectivity,
            "sentiment_assessments": doc._.blob.sentiment_assessments.assessments
        }
    

class SentimentAnalysisHuggingFace:
    def __init__(self, model=None):
        """
        Initialize HuggingFace sentiment analysis pipeline.

        Args:
            model: Optional HuggingFace model name to use.
        """
        try:
            self.model = pipeline('sentiment-analysis', model=model)
        except PipelineException as e:
            raise ValueError(f"Failed to load HuggingFace Pipeline: {e}")

    def analyze(self, text: str) -> list:
        """
        Analyze sentiment using HuggingFace's transformers.

        Args:
            text (str): Input text to analyze.

        Returns:
            list: List of sentiment analysis results with label and score.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
        
        return self.model(text)


class SentimentAnalysis:
    def __init__(self, tool: str = 'nltk', transformer_model=None):
        """
        Initialize sentiment analysis tool.

        Args:
            tool (str): Choose between 'nltk', 'textblob', 'spacy', 'huggingface'.
            transformer_model: Optional model name for HuggingFace model for sentiment analysis.
        """
        if tool == 'nltk':
            self.analyzer = SentimentAnalysisNLTK()
        elif tool == 'textblob':
            self.analyzer = SentimentAnalysisTextBlob()
        elif tool == 'spacy':
            self.analyzer = SentimentAnalysisSpaCy()
        elif tool == 'huggingface':
            if transformer_model is None:
                print("No valid model supplied, default HuggingFace sentiment model will be used.")
            self.analyzer = SentimentAnalysisHuggingFace(model=transformer_model)
        else:
            raise ValueError("Invalid tool selection. Choose from 'nltk', 'textblob', 'spacy', 'huggingface'.")

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment using the selected tool.

        Args:
            text (str): Input text to analyze.

        Returns:
            dict or list: Sentiment analysis result.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
        
        return self.analyzer.analyze(text)
