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
from nltk.tokenize import word_tokenize
from spacy import displacy
import os
import time

import nltk
from nltk.corpus import stopwords
import re
from nltk.corpus import stopwords
import string
import string
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
    

    # def remove_punctuation(text):
    #     # remove punctuation from text
    #     text = text.translate(str.maketrans("", "", string.punctuation))
    #     return text

    def normalize_text(self, text):
                # normalize the text
                tokenize = word_tokenize(text)
                stemmer = PorterStemmer()
                lemmatizer = WordNetLemmatizer()
                text = ' '.join([lemmatizer.lemmatize(stemmer.stem(word)) for word in tokenize])
                #print(text)
                return text
       

    

    def build_ir_system(self, documents, n_components=100):
        # Pré-processamento
        preprocessed_docs = [self.normalize(doc) for doc in documents]
        
        # Vetorização (TF-IDF)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
        
        # Redução de Dimensionalidade (Truncated SVD)
        svd = TruncatedSVD(n_components=n_components)
        reduced_matrix = svd.fit_transform(tfidf_matrix)
        
        # Indexação
        index = {i: reduced_matrix[i] for i in range(len(reduced_matrix))}
        
        return vectorizer, svd, index

    def search(self):
        # Pré-processamento da consulta

        vectorizer, svd, index = self.build_ir_system([document.content for document in self.all_documents])
        preprocessed_query = self.normalize(self.doc)
        
        # Vetorização da consulta
        query_tfidf = vectorizer.transform([preprocessed_query])
        
        # Redução de Dimensionalidade da consulta
        query_reduced = svd.transform(query_tfidf)
        
        # Cálculo da similaridade
        similarities = cosine_similarity(query_reduced, np.array(list(index.values())))
        # Ranking dos documentos
        ranking = np.argsort(similarities[0])[::-1]
        recommendations_with_similarity = [(self.all_documents[idx].title, similarities[0][idx]) 
                                           for idx in ranking if similarities[0][idx] > 0.5]
        
        return recommendations_with_similarity
    
    def separar_palavras(self, texto):
        # Adiciona um espaço antes de cada letra maiúscula que segue uma letra minúscula
        texto = re.sub(r'([a-záéíóúãõâêôàèùç])([A-ZÁÉÍÓÚÃÕÂÊÔÀÈÙÇ])', r'\1 \2', texto)
        # Adiciona um espaço antes de cada número que segue uma letra
        texto = re.sub(r'([a-záéíóúãõâêôàèùç])(\d)', r'\1 \2', texto)
        # Adiciona um espaço antes de cada letra que segue um número
        texto = re.sub(r'(\d)([a-záéíóúãõâêôàèùçA-ZÁÉÍÓÚÃÕÂÊÔÀÈÙÇ])', r'\1 \2', texto)
        return texto

    def lsa_recommender(self):

        documents = [document.semantic_relation["entity"] + " " + document.semantic_relation['verb'] + " " + document.semantic_relation['list_relations'] for document in self.all_documents]

        normalized_documents = [self.normalize(document) for document in documents]
        # chamar o modelo de recomendação por LSA
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(normalized_documents)  # Convert the list of documents into a string

        svd = TruncatedSVD(n_components=100)
        X_svd = svd.fit_transform(X)

        index = {document.title: i for i, document in enumerate(self.all_documents)}

        if isinstance(self.doc, DocumentModel):
            chosen_document = self.doc.semantic_relation['verb'] + ' ' + self.doc.semantic_relation['list_relations']
            chosen_document = self.normalize(chosen_document)
        else:
            chosen_document = self.doc
            chosen_document = self.normalize(chosen_document)
        # print(normalized_documents)
        # print(chosen_document)
        # Transform the chosen document using the same vectorizer and SVD
        chosen_document_vector = vectorizer.transform([chosen_document])
        chosen_document_vector_svd = svd.transform(chosen_document_vector)

        # Compute cosine similarities

        similarities = cosine_similarity(chosen_document_vector_svd, X_svd)

        recommendations = [document.title for document, similarity in zip(self.all_documents, similarities[0]) if similarity > 0.7]
        recommendations_with_similarity = [(document.title, similarity) for document, similarity in zip(self.all_documents, similarities[0]) if similarity > 0.5]
        recommendations_with_similarity.sort(key=lambda x: x[1], reverse=True)
        #recommendations = [title for title, _ in recommendations_with_similarity]
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

        normalized_documents = [self.normalize(document) for document in documents]


        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(normalized_documents)

        
        svd = TruncatedSVD(n_components=100)
        X_svd = svd.fit_transform(X)

        chosen_document = self.text
        chosen_document = self.normalize(chosen_document)
        chosen_document_vector = vectorizer.transform([chosen_document])
        chosen_document_vector_svd = svd.transform(chosen_document_vector)

        similarities = cosine_similarity(chosen_document_vector_svd, X_svd)

        recommendations = [document.title for document, similarity in zip(self.all_documents, similarities[0]) if similarity > 0.7]
        recommendations.sort(reverse=True)
        recommendations_with_similarity = [(document.title, similarity) for document, similarity in zip(self.all_documents, similarities[0]) if similarity > 0.5]
        recommendations_with_similarity.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def all_recommenders_content(self):
        lsa_recommendations = self.lsa_recommender_content()

        dic_recommendations = {'lsa_recommendations': lsa_recommendations}
        return dic_recommendations
    
       
    def accuracy(self, recommendations, chosen_document_title):
        # calcular a acurácia do modelo
        
        return chosen_document_title in recommendations
    
    def normalize_documents(self):

        documents = [document.content for document in self.all_documents[:2]]

        # normalize documents

        # remove stopwords
        stop_words = set(stopwords.words('portuguese'))
        normalized_documents = []
        for document in documents:
            document = ' '.join([word for word in document.split() if word.lower() not in stop_words])
            normalized_documents.append(document)
        
        # remove punctuation
        normalized_documents = [document.translate(str.maketrans("", "", string.punctuation)) for document in normalized_documents]
        # stemming

        ps = PorterStemmer()
        normalized_documents = [' '.join([ps.stem(word) for word in document.split()]) for document in normalized_documents]

        
        print(normalized_documents)
        # normalizar os documentos
        
        return 
    

    def normalize(self, text):
        # normalize the text
        stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stopwords])
        #tokenize = word_tokenize(text)
        # stemmer = PorterStemmer()
        # lemmatizer = WordNetLemmatizer()
        # text = ' '.join([lemmatizer.lemmatize(stemmer.stem(word)) for word in text.split()])
        return text