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
class SemanticRelationRecommender:
    def __init__(self, text: str, title: str, classification:str, dates: str, resolution: str, pdf_path: str):
        self.text = text
        self.title = title
        self.resolution = resolution
        self.pdf_path = pdf_path
        self.dates = dates
        self.classification = classification

    
    def recommender(self):
        # chamar o modelo de recomendação semântica
        pass
    
    def text_preprocessing(self, text):
        # preprocessar o texto

        doc = nlp(text)
        unwanted_words = {"art", "uft", "edu", "br"}

        unecessary_words = [
            "fundação",
            "universidade",
            "federal",
            "tocantins",
            "art",
            "°"
            "FUNDAÇÃO",
            "UNIVERSIDADE",
            "FEDERAL",
            "TOCANTINS",
            "Art",
            "secretaria",
            ".",
            "°",
            "º",
            "conselho",
            "universitario",
            "ensino",
            "pesquisa",
            "extensao",
            "reitor",
            "vice-reitor",
            "edu", 
            "br",
            "uft", 
            "https"
        ]
        import nltk

        # baixa as stopwords
        # nltk.download('stopwords')

        # para escolher as stopwords do português adicionamos a opçaõ de língua "portuguese"
        stopwords = nltk.corpus.stopwords.words('portuguese')
       # pattern = r"[Nn]\s?[º°]\s?\d+\s*/\s*\d+"
        pattern_art = r"\b[aA][rR][tT]\b"
        pattern_uft = r"\b[uU][fF][tT]\b"
        pattern_edu = r"\b[eE][dD][uU]\b"
        
        texto_sem_art = re.sub(pattern_art, "", text)
        texto_sem_uft = re.sub(pattern_uft, "", text)
        texto_sem_edu = re.sub(pattern_edu, "", text)



      
        text_formated = spacy.tokens.Doc(doc.vocab,

         words=[
            token.text.lower()
            for token in doc
            if not token.is_punct
            and not token.is_space
            and not token.is_bracket
            and not token.text in unecessary_words
            and not token.text in stopwords
            and token.text.lower() not in unwanted_words

        ]
        )
        
        text_formatted = " ".join([token.text for token in text_formated if not token.is_space and not token.is_bracket  and not token.is_punct and not token.text in unecessary_words and not token.text in stopwords  and token.text.lower() not in unwanted_words
])
        return text_formatted

    def cut_text(self, value: int):
        text = self.text_preprocessing(self.text)
        palavras = text.split()[:value]
        primeiras_30_linhas = ' '.join(palavras)
        #print(primeiras_30_linhas)
        return primeiras_30_linhas


    def extract_text_verbs(self, doc: spacy.tokens.Doc, verbos_interesse: list) -> str:
        for token in doc:

            #print(token.text, token.pos_)
            if token.pos_ == "VERB" and token.text.lower() in verbos_interesse:
                
                indice_verbo = token.i
                indice_final_sentenca = token.sent[-1].i
                limite_palavras_apos_verbo = 20

                indices_palavras_apos_verbo = [i + 1 for i in range(indice_verbo, indice_final_sentenca) if i + 1 <= indice_final_sentenca][:limite_palavras_apos_verbo]
              
                tokens_entre_verbo_e_palavras = [doc[i] for i in indices_palavras_apos_verbo]
                tokens_entre_verbo_e_palavras.insert(0, doc[indice_verbo])
                texto_entre_verbo_e_palavras = ' '.join(token.text for token in tokens_entre_verbo_e_palavras)
                return texto_entre_verbo_e_palavras

    def text_(self):
        doc = nlp(self.cut_text(90))
        verbos_interesse = ["regulamenta", "dispõe","disp õe" "cria", "estabelece", 
                    "institui", "aprova", "altera",   "anula", 
                    "convoca", "instituir"]
        extract = self.extract_text_verbs(doc, verbos_interesse)
        if not extract:
            doc = nlp(self.cut_text(200))
            extract_more = self.extract_text_verbs(doc, verbos_interesse)
            return extract_more
        return extract 


    def simple_relation_extraction(self):
  
       # resolution = self.resolution if self.resolution else "none"

        verbos_interesse = ["regulamenta", "dispõe", "cria", "disp õe", "estabelece",
                              "institui", "aprova", "altera", "anula", 
                                "convoca", "instituir"]
        if self.text_():
            doc = nlp(self.text_()) 
            #breakpoint()
            for token in doc:
                if token.pos_ == "VERB" and token.text.lower() in verbos_interesse:
                    verb = token.text
                
                    objeto_direto = [child.text for child in token.children if child.dep_ == "obj"]
                    objeto_direto = ' '.join(objeto_direto)
            

            return  verb, objeto_direto
        else:
            return "None", "None"
    
    # def word_cloud(self):
    #     # chamar a função de nuvem de palavras
    #     wordcloud = WordCloud(width=800, height=800, 
    #                 background_color ='white', 
    #                 min_font_size = 10).generate(self.text_preprocessing())
    #     plt.figure(figsize = (8, 8), facecolor = None)
    #     plt.imshow(wordcloud, interpolation="bilinear")
    #     plt.axis("off")
    #     plt.show()

    def complex_relation_extraction(self):
        # extrair relações complexas
        lista = ""
        resolution = self.resolution
        if self.text_():
            doc = nlp(self.text_()) 

            verbos_interesse = ["regulamenta", "dispõe", "cria", "disp õe", "estabelece", 
                        "institui", "aprova", "altera", "anula", 
                        "convoca", "instituir"]
            for token in doc:
                if token.pos_ == "VERB" and token.text.lower() in verbos_interesse:
                    verb = token.text
                    # objeto_direto = [child.text for child in token.children if child.dep_ == "obj"]
                    # objeto_direto = ' '.join(objeto_direto)
                for child in token.children:
                    if child.dep_ == "obj":
                    #print(child.text, child.dep_)
                        lista += child.text + " "
                    if child.dep_ == "obl":
                        lista += child.text + " "
                    if child.dep_ == "nmod":
                        lista += child.text + " "
                    if child.dep_ == "amod":
                        lista += child.text + " "
            
            return verb, lista
        else:
            return "None", "None"

    def semantic_relations_(self):
        # chamar a função de extração de relações semânticas
        # self.syntatic_analysis()
        verb = self.simple_relation_extraction()[0]
        objeto = self.simple_relation_extraction()[1]
        complex_relation = self.complex_relation_extraction()[1]
        title = " ".join([token.text for token in nlp(self.title) if not token.is_stop and not token.is_punct])
        semantic = {
                "entity": title,
                "verb": verb,
                "object": objeto,
                "list_relations": complex_relation
        }
        # self.word_cloud()
        return semantic