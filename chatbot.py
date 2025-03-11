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
        self.selected_theme = None
        self.theme_embeddings = []
        self.presentation = f'Hi, my name is {self.name}, I am a chatbot. I am here to help you with any questions you may have regarding: '
        self.current_embedding = None
        self.nlp = None
        self.w2v = None 

        self.load_models()
        
    def load_models(self):
        
        print('Building Spacy')
        self.nlp = spacy.load("en_core_web_sm")

        print('Building Word2Vec')
        pretrained_model_path = api.load('word2vec-google-news-300', return_path=True)
        self.w2v = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)

        with open("data/categories.json", "r") as json_file:
            self.themes = json.load(json_file)

        for theme in self.themes:

            temp_df = pd.read_csv(f"word2vec_data/{theme}.csv")
            self.dataframes[theme] = temp_df
            self.theme_embeddings.append(self.w2v[theme])

    
    def get_dialogue(self):

        print(f"{self.name}: {self.presentation + ', '.join(self.themes)}.\nPlease ask me a question:")
        
        while True:

            query = input("\nUser: ")

            if query == '':
                self.selected_theme = None
                self.current_embedding = None
                break

            print(f"\nUser: {query}")

            if any(keyword in query.lower() for keyword in ['bye', 'goodbye', 'see you later']):
                print(f"\n{self.name}: Bye Bye")
                self.selected_theme = None
                self.current_embedding = None
                break

            if self.selected_theme is None:
                matching_words =  self.find_matching_words(query)
                self.selected_theme = matching_words[0] if len(matching_words) > 0 else None

                if len(matching_words) > 1:
                    print(f'\n{self.name}: Looks like you are asking about multiple topics. We will solve one topic at time to avoid confusion. We start with the first topic: {matching_words[0]}.')

            if self.selected_theme is None:
                self.selected_theme = self.themes[np.argmax(cosine_similarity(self.embed_sentence(query).reshape(1, -1), self.theme_embeddings))]

             
            answer = self.find_best_answer(query, self.selected_theme)
            print(f"\n{self.name}: {answer}")

        
    def tokenize_sentence(self, sentence):
        
        doc = self.nlp(sentence)
        tokens = [token.text.lower() 
                  for token in doc 
                  if not token.is_punct and not token.is_stop and token.is_alpha and token.pos_ in ["NOUN", 'VERB']]
        return tokens


    def embed_sentence(self, sentence):
        # Split the sentence into tokens
        tokens = self.tokenize_sentence(sentence)
        
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
        
        embed_query = self.embed_sentence(query).reshape(1, -1)

        if self.current_embedding is None:
            self.current_embedding = embed_query
                
        df_embeddings = np.load(f"word2vec_data/{topic}_embeddings.npy")

        most_similar_responce = calculate_similarity_indices((0.3*self.current_embedding + 0.7*embed_query), df_embeddings)

        # Select the best answer
        predicted_sentence = self.dataframes[self.selected_theme].loc[most_similar_responce[0], 'answer']

        # Drop the row from the dataframe associated to the selected anser, such that it cannot be repeated again in the future
        self.dataframes[self.selected_theme] = self.dataframes[self.selected_theme].drop(index=most_similar_responce[0])

        self.current_embedding = 0.7 * (self.embed_sentence(predicted_sentence).reshape(1, -1)*0.5 + embed_query*0.5) + 0.3 * self.current_embedding
        
        return predicted_sentence
        
