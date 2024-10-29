import os
import nltk
import spacy

def download_nltk_model_files(model="punkt"):
    # Check if NLTK models are already downloaded
    nltk_data_path = os.path.expanduser('~') + f'/nltk_data/tokenizers/{model}'
    if not os.path.exists(nltk_data_path):
        print(f"Downloading NLTK '{model}' model")
        nltk.download(model)

    
    nltk_pos_path = os.path.expanduser('~') + '/nltk_data/taggers/averaged_perceptron_tagger'
    if not os.path.exists(nltk_pos_path):
        print("Downloading NLTK 'averaged_perceptron_tagger' model")
        nltk.download('averaged_perceptron_tagger')

def download_spacy_model_files(model="en_core_web_sm"):
    # Check if spaCy model is already downloaded
    try:
        nlp = spacy.load(model)
    except OSError:
        print("Downloading spaCy '{model}' model")
        spacy.cli.download(model)