# Social Tools

## Overview

The **Social Tools** library provides a unified interface for interacting with various social analysis tools, including sentiment analysis, toxicity detection, emotion detection, and other natural language processing (NLP) models. With this library, developers can quickly analyze social media text, chat messages, or other forms of unstructured data for toxicity, sentiment, emotions, and more.

### Key Features

- **Sentiment Analysis**: Determine whether the text expresses positive, negative, or neutral sentiment.
- **Toxicity Detection**: Identify toxic language, hate speech, offensive comments, and inappropriate content.
- **Emotion Detection**: Recognize emotions such as happiness, sadness, anger, and more.
- **Custom NLP Models**: Integrate additional NLP models for detecting bias, misinformation, and other social signals.

## Installation

To install Social Tools, run:

```bash
pip install social-tools
```

## Usage

### Import the Modules

```python
from social_tools import EmotionDetection, SentimentAnalysis, ToxicityDetection
```

## Toxicity Detection

The **ToxicityDetection** module allows you to analyze text for toxic comments using pre-trained models like HuggingFace's `unitary/toxic-bert`.

```python
# Using the unitary/toxic-bert transformer model
tox_detector = ToxicityDetection(tool='transformer', model='unitary/toxic-bert')
result = tox_detector.analyze("I hate you.")
print(result)
```

This returns:

```python
[{'label': 'toxic', 'score': 0.9475088119506836}]
```

You can also analyze multiple texts at once:

```python
tox_detector.analyze(["I hate you.", "This is harsh"])
```

Output:

```python
[
    {'label': 'toxic', 'score': 0.9475088119506836},
    {'label': 'toxic', 'score': 0.002488125581294298}
]
```

## Sentiment Analysis

The **SentimentAnalysis** module offers several options for analyzing the sentiment of text, including NLTK, SpaCy, and HuggingFace models.

```python
# Using NLTK
sa = SentimentAnalysis(tool='nltk')
result = sa.analyze("This is awesome!")
print(result)
```

Output:

```python
[{'neg': 0.0, 'neu': 0.313, 'pos': 0.687, 'compound': 0.6588}]
```

### Using SpaCy:

```python
sa = SentimentAnalysis(tool='spacy')
result = sa.analyze("This is awesome!")
print(result)
```

Output:

```python
[{
    'polarity': 1.0, 
    'subjectivity': 1.0, 
    'sentiment_assessments': [(['awesome', '!'], 1.0, 1.0, None)]
}]
```

### Using HuggingFace Transformer:

```python
sa = SentimentAnalysis(tool='huggingface')
result = sa.analyze("This is awesome!")
print(result)
```

Output:

```python
[{'label': 'POSITIVE', 'score': 0.9998669624328613}]
```

#### Custom HuggingFace Models

You can specify a custom HuggingFace transformer model by passing the model name during initialization:

```python
sa = SentimentAnalysis(tool='huggingface', transformer_model="cardiffnlp/twitter-roberta-base-sentiment-latest")
result = sa.analyze("This is awesome!")
print(result)
```

Output:

```python
[{'label': 'positive', 'score': 0.9813949465751648}]
```

## Emotion Detection

The **EmotionDetection** module allows you to detect emotions such as happiness, sadness, and anger. For example, using HuggingFace models:

```python
emotion_detector = EmotionDetection(tool='huggingface')
result = emotion_detector.analyze("I am so happy today!")
print(result)
```

This will return:

```python
[{'label': 'joy', 'score': 0.95}]
```

## Flexible Input Handling

All `analyze` functions in each module accept both single strings (`str`) and lists of strings (`List[str]`) as input:

```python
# Single input
result = sa.analyze("I love this!")

# Multiple inputs
result = sa.analyze(["I love this!", "This is terrible."])
```

### HuggingFace Transformer Parameters

When using a HuggingFace transformer model, you can pass additional parameters during initialization, such as `return_all_scores`:

```python
sa = SentimentAnalysis(tool='huggingface', transformer_model="bert-base-uncased", return_all_scores=True)
result = sa.analyze("This is fantastic!")
print(result)
```

## Conclusion

The Social Tools library simplifies the process of analyzing social data by providing multiple sentiment, emotion, and toxicity detection tools in a unified interface. You can integrate popular NLP libraries like NLTK, SpaCy, and HuggingFace models into your workflow seamlessly.

For more information about the supported HuggingFace models and additional parameters, refer to the [HuggingFace documentation](https://huggingface.co/models).

## Acknowledgements

This project would not have been possible without the contributions of the following open-source projects:

- **[Detoxify](https://github.com/unitaryai/detoxify)**: For providing pre-trained models to detect toxic content in text.
- **[HuggingFace](https://huggingface.co/)**: For providing a wide variety of pre-trained transformer models and their powerful `transformers` library.
- **[NLTK (Natural Language Toolkit)](https://www.nltk.org/)**: For providing robust tools for text processing and sentiment analysis.
- **[SpaCy](https://spacy.io/)**: For offering fast and efficient NLP capabilities, along with the `spacytextblob` extension for sentiment analysis.
- **[TextBlob](https://textblob.readthedocs.io/en/dev/)**: For providing an easy-to-use interface for text processing, sentiment analysis, and other NLP tasks.

A huge thank you to these projects and their respective communities for building the foundational tools that made this library possible.

## Citation

if you use the detoxify  module in this tool, kindly cite

```markdown
@misc{Detoxify,
  title={Detoxify},
  author={Hanu, Laura and {Unitary team}},
  howpublished={Github. https://github.com/unitaryai/detoxify},
  year={2020}
}
```
