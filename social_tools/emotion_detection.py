from transformers import pipeline
from textblob import TextBlob
import logging
from typing import Union, List

# Set up logging for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDetectionHuggingFace:
    def __init__(self, model: str = 'j-hartmann/emotion-english-distilroberta-base', **kwargs):
        """
        Initialize HuggingFace emotion detection model.

        Args:
            model (str): Name of the HuggingFace model to use. Default is 'j-hartmann/emotion-english-distilroberta-base'.
            kwargs: Additional arguments to pass to the HuggingFace pipeline.
        """
        try:
            self.model = pipeline('text-classification', model=model, **kwargs)
            logger.info(f"Emotion model '{model}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace model: {e}")
            raise ValueError(f"Failed to initialize HuggingFace model: {e}")

    def analyze(self, text: Union[str, List[str]]) -> list:
        """
        Analyze text to detect emotions using HuggingFace transformers.

        Args:
            text (str, List[str]): Input text or list of texts to analyze.

        Returns:
            list: List of detected emotions with confidence scores.
        """
        if isinstance(text, str):
            text = [text]  # Convert single string to list for consistency
        elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("Input must be a non-empty string or a list of non-empty strings.")

        try:
            result = self.model(text)
            logger.info("Emotion detection with HuggingFace successful.")
            return result
        except Exception as e:
            logger.error(f"Error during HuggingFace emotion detection: {e}")
            raise ValueError(f"Emotion detection failed: {e}")


class EmotionDetectionTextBlob:
    def analyze(self, text: Union[str, List[str]]) -> List[dict]:
        """
        Analyze emotion using TextBlob (via polarity/subjectivity as a proxy for emotions).

        Args:
            text (str, List[str]): Input text or list of texts to analyze.

        Returns:
            List[dict]: List of basic analysis results with polarity and subjectivity as proxies for emotions.
        """
        if isinstance(text, str):
            text = [text]  # Convert single string to list for consistency
        elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("Input must be a non-empty string or a list of non-empty strings.")

        results = []
        for t in text:
            blob = TextBlob(t)
            results.append({
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            })
        return results


class EmotionDetection:
    def __init__(self, tool: str = 'huggingface', **kwargs):
        """
        Initialize emotion detection tool.

        Args:
            tool (str): Choose between 'huggingface', 'textblob'.
            kwargs: Arguments specific to the chosen tool.
        """
        try:
            if tool == 'huggingface':
                self.analyzer = EmotionDetectionHuggingFace(**kwargs)
            elif tool == 'textblob':
                self.analyzer = EmotionDetectionTextBlob()
            else:
                raise ValueError("Invalid tool selection. Choose 'huggingface' or 'textblob'.")
            logger.info(f"Emotion detection tool '{tool}' initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing emotion detection tool: {e}")
            raise ValueError(f"Failed to initialize emotion detection tool: {e}")

    def analyze(self, text: Union[str, List[str]]) -> Union[dict, List[dict]]:
        """
        Analyze emotion using the selected tool.

        Args:
            text (str, List[str]): Input text or list of texts to analyze.

        Returns:
            Union[dict, List[dict]]: Emotion analysis result (list if multiple texts are passed).
        """
        try:
            result = self.analyzer.analyze(text)
            logger.info("Emotion detection successful.")
            return result
        except Exception as e:
            logger.error(f"Error during emotion detection: {e}")
            raise ValueError(f"Emotion detection failed: {e}")
