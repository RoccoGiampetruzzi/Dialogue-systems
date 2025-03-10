import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import spacy
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors

nlp = spacy.load("en_core_web_sm")
pretrained_model_path = api.load('word2vec-google-news-300', return_path=True)
pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)


def calculate_similarity_indices(vector1, vector2):
    # Calculate the cosine similarity between each row in vector1 and vector2
    similarities = cosine_similarity(vector1, vector2)

    # For each row in vector1, find the index of the most similar row in vector2 
    most_similar_indices = np.argmax(similarities, axis=1)
    
    return most_similar_indices


def tokenize_sentence(sentence):
    
    doc = nlp(sentence)
    tokens = [token.text.lower() for token in doc if not token.is_punct]
    return tokens


def embed_sentence(sentence):
    # Split the sentence into tokens
    tokens = tokenize_sentence(sentence)
    
    # Initialize an empty array to store the word embeddings
    embeddings = np.zeros(pretrained_model.vector_size)
    
    n = 0
    # Iterate over each token in the sentence
    for token in tokens:
        # Check if the token is present in the pretrained word2vec model
        if token in pretrained_model:
            # Add the word embedding to the sentence embeddings
            embeddings += pretrained_model[token]
            n+=1
    
    # Normalize the sentence embeddings
    embeddings /= n+1
    
    return embeddings