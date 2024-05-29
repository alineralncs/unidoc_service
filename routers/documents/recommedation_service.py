from models.document import Document as DocumentModel
import spacy
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
import nltk
import numpy as np
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging
from docarray import Document, DocumentArray
import spacy
import pprint
from spacy import displacy
import os
import time

import nltk
from nltk.corpus import stopwords
import re
nlp = spacy.load("pt_core_news_lg")
class DocumentModel:
    def __init__(self, content, title, resolution, pdf_path, date, status, semantic_relation, counsil):
        self.content = content
        self.title = title
        self.resolution = resolution
        self.pdf_path = pdf_path
        self.date = date
        self.status = status
        self.semantic_relation = semantic_relation
        self.counsil = counsil

class RecommendationService:
    def __init__(self, doc: Union[DocumentModel, str], all_documents: list):
        self.all_documents = all_documents
        
        if isinstance(doc, DocumentModel):
            self.doc = doc
            self.text = doc.content
            self.title = doc.title
            self.resolution = doc.resolution
            self.pdf_path = doc.pdf_path
            self.dates = doc.date
            self.classification = doc.status
            self.semantic_relation = doc.semantic_relation
            self.counsil = doc.counsil
        elif isinstance(doc, str):
            self.doc = doc
            self.text = doc
            self.title = ""
            self.resolution = ""
            self.pdf_path = ""
            self.dates = ""
            self.classification = ""
            self.semantic_relation = ""
            self.counsil = ""


        
    def calculate_similarity(self, chosen_document_semantic_relation, document_semantic_relation):
        # calcular a similaridade entre os documentos

        nlp = spacy.load("pt_core_news_sm")
        doc1 = nlp(chosen_document_semantic_relation)
        doc2 = nlp(document_semantic_relation)

        similarity = doc1.similarity(doc2)
        return similarity

    def similarity_recommender(self):
        # chamar o modelo de recomendação semântica
        chosen_document = self.doc
        chosen_document_semantic_relation = chosen_document.semantic_relation['list_relations']
        chosen_document_title = chosen_document.title

        chosen_document_semantic_relation_verb = chosen_document.semantic_relation['verb']
        chosen_document_semantic_relation_obj = chosen_document.semantic_relation['object']

        recommendations = []

        for document in self.all_documents:
            if document.title != chosen_document_title:
                document_semantic_relation = document.semantic_relation['list_relations']
                document_semantic_relation_verb =  document.semantic_relation['verb']
                document_semantic_relation_obj =  document.semantic_relation['object']

                print('chosen::: ',chosen_document_semantic_relation)
                print(document_semantic_relation)

            
                similarity = self.calculate_similarity(chosen_document_semantic_relation, document_semantic_relation)
                similarity_verb = self.calculate_similarity(chosen_document_semantic_relation_verb, document_semantic_relation_verb)
                similarity_obj = self.calculate_similarity(chosen_document_semantic_relation_obj, document_semantic_relation_obj)


                if similarity > 0.5 and similarity_verb > 0.5 and similarity_obj > 0.5:
                    print('title', document.semantic_relation)
                    print('similarity::: ', similarity, similarity_verb, similarity_obj)
                    
                    recommendations.append((document.title, similarity, similarity_verb, similarity_obj))

        recommendations.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        recommendations = [title for title, _, _, _ in recommendations]

        return recommendations
        

    def embeddings_recommender(self):
        # chamar o modelo de recomendação por embeddings
        nltk.download('stopwords')
        stop_words = set(stopwords.words('portuguese'))
        sentences = []

        chosen_token = [self.semantic_relation['verb']] + self.semantic_relation['list_relations'].split()
        
        # treinar o modelo word2vec com a lista de semantic relations
        tokens = []
        for document in self.all_documents:
            tokens = [document.semantic_relation['verb']] + document.semantic_relation['list_relations'].split()
            sentences.append(tokens)
        
        preprocessed_sentences = [simple_preprocess(' '.join(sentence)) for sentence in sentences]

        model = Word2Vec(sentences=preprocessed_sentences, vector_size=100, window=5, min_count=1, workers=4)

        model.save("word2vec.model")

        model = Word2Vec.load("word2vec.model")

        
        chosen_document_embeddings = model.wv[chosen_token].reshape(1, -1)  # Reshape chosen_document_embeddings to have the same number of rows as document_embeddings
        similarities = []

        for document in self.all_documents:
            document_embeddings = model.wv[[document.semantic_relation['verb']] + document.semantic_relation['list_relations'].split()]
            similarity = np.dot(chosen_document_embeddings, document_embeddings.T) / (np.linalg.norm(chosen_document_embeddings) * np.linalg.norm(document_embeddings))
            similarities.append(similarity)

        recommendations = [document.title for document, similarity in zip(self.all_documents, similarities) if similarity > 0.5]
        recommendations.sort(reverse=True)

        return recommendations
    

    

    def lsa_recommender(self):

        documents = [document.semantic_relation["entity"] +document.semantic_relation['verb'] + document.semantic_relation['list_relations'] for document in self.all_documents]
        # chamar o modelo de recomendação por LSA
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(documents)  # Convert the list of documents into a string

        svd = TruncatedSVD(n_components=100)
        X_svd = svd.fit_transform(X)


        if isinstance(self.doc, DocumentModel):
            chosen_document = self.doc.semantic_relation['verb'] + ' ' + self.doc.semantic_relation['list_relations']
        else:
            chosen_document = self.doc

        # Transform the chosen document using the same vectorizer and SVD
        chosen_document_vector = vectorizer.transform([chosen_document])
        chosen_document_vector_svd = svd.transform(chosen_document_vector)

        # Compute cosine similarities
        similarities = cosine_similarity(chosen_document_vector_svd, X_svd)

        recommendations = [document.title for document, similarity in zip(self.all_documents, similarities[0]) if similarity > 0.5]
        recommendations.sort(reverse=True)
        return recommendations
    
    def all_recommenders(self):
        # chamar todos os modelos de recomendação
        #similarity_recommendations = self.similarity_recommender()
        #embeddings_recommendations = self.embeddings_recommender()
        lsa_recommendations = self.lsa_recommender()
        # 'similarity_recommendations': similarity_recommendations, 

        dic_recommendations = {'lsa_recommendations': lsa_recommendations}
        
        return dic_recommendations
    






    def lsa_recommender_content(self):

        documents = [document.content for document in self.all_documents]
        



        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(documents)

        svd = TruncatedSVD(n_components=100)
        X_svd = svd.fit_transform(X)

        chosen_document = self.text
        chosen_document_vector = vectorizer.transform([chosen_document])
        chosen_document_vector_svd = svd.transform(chosen_document_vector)

        similarities = cosine_similarity(chosen_document_vector_svd, X_svd)

        recommendations = [document.title for document, similarity in zip(self.all_documents, similarities[0]) if similarity > 0.5]
        recommendations.sort(reverse=True)
        return recommendations
    
    def all_recommenders_content(self):
        lsa_recommendations = self.lsa_recommender_content()

        dic_recommendations = {'lsa_recommendations': lsa_recommendations}
        return dic_recommendations
    
       
    def accuracy(self, recommendations, chosen_document_title):
        # calcular a acurácia do modelo
        
        return chosen_document_title in recommendations
    