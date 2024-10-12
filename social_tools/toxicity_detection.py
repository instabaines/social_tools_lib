import detoxify
from transformers import pipeline

from typing import List, Union
import logging


# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetoxifyWrapper:
    def __init__(self, model: str = 'original', **kwargs):
        """
        Initialize Detoxify with the selected model and optional parameters.

        Args:
            model (str): Choose the Detoxify model variant. Default is 'original'.
            kwargs: Additional parameters for the Detoxify model.
        """
        try:
            self.model = detoxify.Detoxify(model, **kwargs)
            logger.info(f"Detoxify model '{model}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing Detoxify model: {e}")
            raise ValueError(f"Failed to initialize Detoxify model: {e}")

    def analyze(self, text: str) -> dict:
        """
        Analyze text using Detoxify.

        Args:
            text (str): Input text to analyze.

        Returns:
            dict: Toxicity analysis result with scores.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
        
        try:
            result = self.model.predict(text)
            logger.info("Detoxify analysis successful.")
            return result
        except Exception as e:
            logger.error(f"Error in Detoxify analysis: {e}")
            raise ValueError(f"Detoxify analysis failed: {e}")
    


class ToxicityTransformer:
    def __init__(self, model: str = 'unitary/toxic-bert', **kwargs):
        """
        Initialize HuggingFace toxicity pipeline.

        Args:
            model (str): Name of the HuggingFace model to use. Default is 'unitary/toxic-bert'.
            kwargs: Additional arguments to pass to the HuggingFace pipeline.
        """
        try:
            self.model = pipeline('text-classification', model=model, **kwargs)
            logger.info(f"Transformer model '{model}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace pipeline: {e}")
            raise ValueError(f"Failed to initialize HuggingFace pipeline: {e}")

    def analyze(self, text: str) -> list:
        """
        Analyze text using a HuggingFace transformer model.

        Args:
            text (str): Input text to analyze.

        Returns:
            list: List of toxicity analysis results with labels and scores.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        try:
            result = self.model(text)
            logger.info("Transformer analysis successful.")
            return result
        except Exception as e:
            logger.error(f"Error in Transformer analysis: {e}")
            raise ValueError(f"Transformer analysis failed: {e}")


class ToxicityDetection:
    def __init__(self, tool: str = 'detoxify', **kwargs):
        """
        Initialize toxicity detection tool.

        Args:
            tool (str): Choose between 'detoxify', and 'transformer'.
            kwargs: Arguments specific to the chosen tool.
        """
        try:
            if tool == 'detoxify':
                self.analyzer = DetoxifyWrapper(**kwargs)
            elif tool == 'transformer':
                self.analyzer = ToxicityTransformer(**kwargs)
            else:
                raise ValueError("Invalid tool selection. Choose from  'detoxify', 'transformer'.")
            logger.info(f"Toxicity detection tool '{tool}' initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing toxicity detection tool: {e}")
            raise ValueError(f"Failed to initialize toxicity detection tool: {e}")

    def analyze(self, text: Union[str, List[str]]) -> dict:
        """
        Analyze toxicity using the selected tool.

        Args:
            text (Union[str, List[str]]): Input text to analyze.

        Returns:
            dict or list: Toxicity analysis result.
        """
        try:
            result = self.analyzer.analyze(text)
            logger.info("Toxicity analysis successful.")
            return result
        except Exception as e:
            logger.error(f"Error during toxicity analysis: {e}")
            raise ValueError(f"Toxicity analysis failed: {e}")