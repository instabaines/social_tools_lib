import pytest
from social_tools.toxicity_detection import ToxicityDetection


# Test case: Initialize and analyze using Detoxify model
def test_detoxify_basic_analysis():
    tox_detector = ToxicityDetection(tool='detoxify', model='unbiased')
    result = tox_detector.analyze("You are an idiot!")
    
    assert isinstance(result, dict), "Detoxify should return a dictionary."
    assert 'toxicity' in result, "Result should contain a 'toxicity' score."


# Test case: Initialize and analyze using Transformer model
def test_transformer_basic_analysis():
    tox_detector = ToxicityDetection(tool='transformer', model='unitary/toxic-bert')
    result = tox_detector.analyze("I hate you.")
    
    assert isinstance(result, list), "Transformer should return a list."
    assert len(result) > 0, "Transformer result should not be empty."
    assert 'label' in result[0], "Each result should contain a 'label'."
    assert 'score' in result[0], "Each result should contain a 'score'."


# Test case: Handle non-string input
def test_non_string_input():
    tox_detector = ToxicityDetection(tool='detoxify')
    
    with pytest.raises(ValueError) as excinfo:
        tox_detector.analyze(12345)
    
    assert "Input text must be a non-empty string or a list of non-empty strings." in str(excinfo.value), "Should raise ValueError for non-string input."


# Test case: Handle list with invalid elements
def test_invalid_list_elements():
    tox_detector = ToxicityDetection(tool='detoxify')
    
    with pytest.raises(ValueError) as excinfo:
        tox_detector.analyze(["Valid text", 12345])  # Invalid element in list
    
    assert "Toxicity analysis failed: Input text must be a non-empty string or a list of non-empty strings." in str(excinfo.value), excinfo.value



# Test case: Handle model initialization failure
def test_invalid_model_initialization():
    with pytest.raises(ValueError) as excinfo:
        ToxicityDetection(tool='transformer', model='invalid-model-name')  # Invalid HuggingFace model
    
    assert "Failed to initialize HuggingFace pipeline" in str(excinfo.value), "Should raise error for invalid model."


# Test case: Long input string
def test_long_input_string():
    long_text = "You are terrible. " * 100  # Very long string
    tox_detector = ToxicityDetection(tool='detoxify', model='original')
    result = tox_detector.analyze(long_text)
    
    assert isinstance(result, dict), "Detoxify should handle long input strings."
    assert 'toxicity' in result, "Result should contain a 'toxicity' score."


# Test case: Very short input string
def test_short_input_string():
    short_text = "Hi"
    tox_detector = ToxicityDetection(tool='detoxify', model='original')
    result = tox_detector.analyze(short_text)
    
    assert isinstance(result, dict), "Detoxify should handle very short input strings."
    assert 'toxicity' in result, "Result should contain a 'toxicity' score."


# Test case: Edge case with special characters
def test_special_characters():
    special_text = "!@#$%^&*()"
    tox_detector = ToxicityDetection(tool='detoxify', model='original')
    result = tox_detector.analyze(special_text)
    
    assert isinstance(result, dict), "Detoxify should handle special characters."
    assert 'toxicity' in result, "Result should contain a 'toxicity' score."


if __name__ == "__main__":
    pytest.main()
