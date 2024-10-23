import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import pipeline
from transformers.pipelines import PipelineException
import subprocess
import logging
from typing import Union, List
from .utils import download_nltk_model_files,download_spacy_model_files
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class SentimentAnalysisNLTK:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        #download NLTK model if not present
        download_nltk_model_files('vader_lexicon')

    def analyze(self, text: Union[str, List[str]]) -> List[dict]:
        """
        Analyze sentiment using NLTK's VADER model.

        Args:
            text (Union[str, List[str]]): Input text or list of texts to analyze.

        Returns:
            List[dict]: Sentiment analysis result with polarity scores.
        """
        if isinstance(text, str):
            text = [text]  # Convert single string to list for uniform processing
        elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("Input must be a non-empty string or a list of non-empty strings.")

        results = [self.analyzer.polarity_scores(t) for t in text]
        return results


class SentimentAnalysisTextBlob:
    def analyze(self, text: Union[str, List[str]]) -> List[dict]:
        """
        Analyze sentiment using TextBlob.

        Args:
            text (Union[str, List[str]]): Input text or list of texts to analyze.

        Returns:
            List[dict]: Sentiment analysis result with polarity and subjectivity.
        """
        if isinstance(text, str):
            text = [text]  # Convert single string to list for uniform processing
        elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("Input must be a non-empty string or a list of non-empty strings.")

        results = [{'polarity': TextBlob(t).sentiment.polarity, 'subjectivity': TextBlob(t).sentiment.subjectivity} for t in text]
        return results


class SentimentAnalysisSpaCy:
    def __init__(self,model='en_core_web_sm'):
        """
        Initialize SpaCy with the spacytextblob pipeline.
        If the SpaCy model is not available, download it automatically.
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            logger.info(f"SpaCy model {model} not found. Attempting to download it...")
            self.download_spacy_model(model)
            self.nlp = spacy.load(model)

        # Ensure spacytextblob is added to the pipeline
        if "spacytextblob" not in self.nlp.pipe_names:
            self.nlp.add_pipe("spacytextblob")

    def download_spacy_model(self,model):
        """
        Download the SpaCy  model.
        """
        try:
            download_spacy_model_files(model)
            # subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            logger.info(f"Model '{model}' downloaded successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download SpaCy model: {e}")
            raise OSError("SpaCy model was not downloaded. Check SpaCy documentation for downloading models and try again.")

    def analyze(self, text: Union[str, List[str]]) -> List[dict]:
        """
        Analyze sentiment using SpaCy with spacytextblob.

        Args:
            text (Union[str, List[str]]): Input text or list of texts to analyze.

        Returns:
            List[dict]: Sentiment analysis result with polarity and subjectivity.
        """
        if isinstance(text, str):
            text = [text]  # Convert single string to list for uniform processing
        elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("Input must be a non-empty string or a list of non-empty strings.")

        results = []
        for t in text:
            doc = self.nlp(t)
            results.append({
                'polarity': doc._.blob.polarity,
                'subjectivity': doc._.blob.subjectivity,
                "sentiment_assessments": doc._.blob.sentiment_assessments.assessments
            })
        return results


class SentimentAnalysisHuggingFace:
    def __init__(self, model=None, **kwargs):
        """
        Initialize HuggingFace sentiment analysis pipeline.

        Args:
            model: Optional HuggingFace model name to use.
            kwargs: Additional parameters like return_all_scores.
        """
        try:
            self.model = pipeline('sentiment-analysis', model=model, **kwargs)
            logger.info(f"HuggingFace model '{model if model else 'default'}' loaded successfully.")
        except PipelineException as e:
            logger.error(f"Failed to load HuggingFace Pipeline: {e}")
            raise ValueError(f"Failed to load HuggingFace Pipeline: {e}")

    def analyze(self, text: Union[str, List[str]]) -> List[dict]:
        """
        Analyze sentiment using HuggingFace's transformers.

        Args:
            text (Union[str, List[str]]): Input text or list of texts to analyze.

        Returns:
            List[dict]: List of sentiment analysis results with label and score.
        """
        if isinstance(text, str):
            text = [text]  # Convert single string to list for uniform processing
        elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("Input must be a non-empty string or a list of non-empty strings.")

        results = self.model(text)
        return results


class SentimentAnalysis:
    def __init__(self, tool: str = 'nltk', transformer_model=None, **kwargs):
        """
        Initialize sentiment analysis tool.

        Args:
            tool (str): Choose between 'nltk', 'textblob', 'spacy', 'huggingface'.
            transformer_model: Optional model name for HuggingFace sentiment analysis.
            kwargs: Additional arguments for HuggingFace models, like 'return_all_scores'.
        """
        if tool == 'nltk':
            self.analyzer = SentimentAnalysisNLTK()
        elif tool == 'textblob':
            self.analyzer = SentimentAnalysisTextBlob()
        elif tool == 'spacy':
            self.analyzer = SentimentAnalysisSpaCy()
        elif tool == 'huggingface':
            if transformer_model is None:
                logger.info("No valid model supplied, default HuggingFace sentiment model will be used.")
            self.analyzer = SentimentAnalysisHuggingFace(model=transformer_model, **kwargs)
        else:
            raise ValueError("Invalid tool selection. Choose from 'nltk', 'textblob', 'spacy', 'huggingface'.")

    def analyze(self, text: Union[str, List[str]]) -> List[dict]:
        """
        Analyze sentiment using the selected tool.

        Args:
            text (Union[str, List[str]]): Input text or list of texts to analyze.

        Returns:
            List[dict]: Sentiment analysis result.
        """
        return self.analyzer.analyze(text)
