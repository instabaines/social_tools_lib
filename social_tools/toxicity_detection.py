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

    def analyze(self, text: Union[str, List[str]]) -> List[dict]:
        """
        Analyze text using Detoxify.

        Args:
            text (Union[str, List[str]]): Input text or list of texts to analyze.

        Returns:
            List[dict]: Toxicity analysis result with scores for each input.
        """
        if isinstance(text, str):
            text = [text]  # Convert single string to list for consistency
        elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("Input must be a non-empty string or a list of non-empty strings.")

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

    def analyze(self, text: Union[str, List[str]]) -> List[dict]:
        """
        Analyze text using a HuggingFace transformer model.

        Args:
            text (Union[str, List[str]]): Input text or list of texts to analyze.

        Returns:
            List[dict]: List of toxicity analysis results with labels and scores.
        """
        if isinstance(text, str):
            text = [text]  # Convert single string to list for consistency
        elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("Input must be a non-empty string or a list of non-empty strings.")

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
            tool (str): Choose between 'detoxify' and 'transformer'.
            kwargs: Arguments specific to the chosen tool.
        """
        try:
            if tool == 'detoxify':
                self.analyzer = DetoxifyWrapper(**kwargs)
            elif tool == 'transformer':
                self.analyzer = ToxicityTransformer(**kwargs)
            else:
                raise ValueError("Invalid tool selection. Choose from 'detoxify' or 'transformer'.")
            logger.info(f"Toxicity detection tool '{tool}' initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing toxicity detection tool: {e}")
            raise ValueError(f"Failed to initialize toxicity detection tool: {e}")

    def analyze(self, text: Union[str, List[str]]) -> List[dict]:
        """
        Analyze toxicity using the selected tool.

        Args:
            text (Union[str, List[str]]): Input text or list of texts to analyze.

        Returns:
            List[dict]: Toxicity analysis result.
        """
        
        try:
            if isinstance(text, str):
                text = [text]  # Convert single string to list for uniform processing
            elif not isinstance(text, list) or not all(isinstance(t, str) for t in text):
                raise ValueError("Input text must be a non-empty string or a list of non-empty strings.")
            result = self.analyzer.analyze(text)
            logger.info("Toxicity analysis successful.")
            return result
        except Exception as e:
            logger.error(f"Error during toxicity analysis: {e}")
            raise ValueError(f"Toxicity analysis failed: {e}")
