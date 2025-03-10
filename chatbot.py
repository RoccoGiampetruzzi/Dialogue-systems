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
from utils import * 
import json
import gensim.downloader as api

class ChatBot:

    def __init__(self):
        
        self.dataframes = {}
        self.themes = None
        self.name = 'MarioBot'

        self.presentation = 'Hi, my name is MarioBot, I am a chatbot. I am here to help you with any questions you may have regarding: '

        self.nlp = None
        self.w2v = None

    def load_models(self):

        self.nlp = spacy.load("en_core_web_sm")

        pretrained_model_path = api.load('word2vec-google-news-300', return_path=True)
        self.w2v = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)

        with open("data/categories.json", "r") as json_file:
            self.themes = json.load(json_file)

        for theme in self.themes:

            temp_df = pd.read_csv(f"word2vec_data/{theme}.csv")
            self.dataframes[theme] = temp_df

    
    def get_dialogue(self):

        self.load_models()

        print(f"{self.name}: {self.presentation + ', '.join(self.themes)}.\nPlease ask me a question:")
        
        while True:

            query = input("Insert the question: ")

            print(f"\nUser: {query}")

            matching_words =  self.find_matching_words(query)

            topic = matching_words[0]

            if len(matching_words) > 1:
                
                print(f'\n{self.name}: Looks like you are asking about multiple topics. We will solve one topic at time to avoid confusion. We start with the first topic: {matching_words[0]}.')

            elif len(matching_words) == 0:

                print(f"\n{self.name}: I am sorry, I do not have information about that topic. Try to rephrase the question or ask me something else.")

            
            answer = self.find_best_answer(query, topic)
            print(f"\n{self.name}: {answer}")

        
    def tokenize_sentence(self, sentence):
        
        doc = self.nlp(sentence)
        tokens = [token.text.lower() for token in doc if not token.is_punct]
        return tokens


    def embed_sentence(self, sentence):
        # Split the sentence into tokens
        tokens = tokenize_sentence(sentence)
        
        # Initialize an empty array to store the word embeddings
        embeddings = np.zeros(self.w2v.vector_size)
        
        n = 0
        # Iterate over each token in the sentence
        for token in tokens:
            # Check if the token is present in the pretrained word2vec model
            if token in self.w2v:
                # Add the word embedding to the sentence embeddings
                embeddings += self.w2v[token]
                n+=1
        
        # Normalize the sentence embeddings
        embeddings /= n+1
        
        return embeddings


    def find_matching_words(self, query):
        return [word for word in self.themes if word in query]
    
    def find_best_answer(self, query, topic):
        
        embed_query = embed_sentence(query)
        df = self.dataframes[topic]['question_embeddings'].to_numpy()

        most_similar_responce = calculate_similarity_indices(np.array(embed_query).reshape(1, -1) , np.array(df))

        predicted_sentence = df.iloc[most_similar_responce[0]]['answer']

        return predicted_sentence
        