import re
from typing import List, Tuple

import PyPDF2
import spacy
from docarray import DocumentArray
import pprint
import logging

from routers.documents.semantic_service import SemanticRelationRecommender

nlp = spacy.load("pt_core_news_lg")

    # title = Column(String, index=True)
    # pdf_path = Column(String)
    # content = Column(String)
    # created_date = Column(DateTime, default=datetime.datetime.now())
    # status = Column(String)
    # resolution = Column(String)
    # signature = Column(String)
    # date = Column(String)
    # semantic_relation = Column(JSON)
    # counsil = Column(String)


class DocumentService:
    def __init__(self, doc_array: DocumentArray):
        self.doc_array = doc_array
        self.text = ""

    
    def get_pdf_path(self) -> str:
        # extrair o caminho do documento
        for i, document in enumerate(self.doc_array):
            pdf_path = document.tags.get("path")
            return pdf_path
        
    def extract_title(self) -> str:
        # extrair o título do documento
        pdf_path = self.get_pdf_path()
        pattern = r'(?:.*\\){3}(.*)'
        match = re.search(pattern, pdf_path)
        if match:
            title = match.group(1)
            title = title.replace(".pdf", "")
            return title
        else:
            return None
    
    def extract_text(self) -> str:
        for document in self.doc_array:
            pdf_path = document.tags.get("path")
            if pdf_path:
                read_pdf = PyPDF2.PdfReader(pdf_path)
                number_of_pages = len(read_pdf.pages)
                if number_of_pages > 5: # Se o documento tiver mais de 5 páginas, leia apenas as primeiras 5 páginas
                    number_of_pages = 5
                for page_num in range(number_of_pages):
                    page = read_pdf.pages[page_num]
                    content = page.extract_text()
                    doc = nlp(content)
                    self.text += doc.text + "\n"
        
        return self.text
    
    def is_document_legible(self) -> bool:
        # verificar se o documento é legível
        extracted_text = self.extract_text()
        
        if extracted_text and not extracted_text.isspace():
            return True
        return False
            
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

        stopwords = nltk.corpus.stopwords.words('portuguese')
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
    def extract_resolutions(self, text) -> str:
        # extrair as resoluções do documento 
        # pattern = r"N[º°]\s?\d+\s*/\s*\d+"
        # pattern_find_text= r"[Nn]\s?[º°]\s?\d+\s*/\s*\d+"

        pattern_from_title = r'^(\d+)-\d+'  # pattern para extrair resolução do título
        # pattern_ = r"N[º°]\s\d{3}/\d{4}"
        title = self.extract_title()
        # pattern_art = r"\b[aA][rR][tT]\b"
        # pattern_uft = r"\b[uU][fF][tT]\b"
        # pattern_edu = r"\b[eE][dD][uU]\b"

        
        #pattern = r"[Nn]\s*[º°]\s{2,}\d{0,2}\s*/\s*\d+"
        
        match = re.search(pattern_from_title, title.lower(), re.IGNORECASE)
        if match:
            #print('match', match.group())
            resolution_number = match.group()
            resolution_number = resolution_number.replace("-", "/")
            return resolution_number
        else:  
            match = re.search(r"\d{3}/\d{4}", self.text)
            if match:
                resolution_number = match.group()
                return resolution_number
            return None
        
    def extract_signatures(self, text) -> str:
        # extrair as assinaturas do documento

        # prof_pattern = r"prof\s+([a-z]+\s+[a-z]+)\s+presidente"

        # reitor_pattern = r"[A-Z][A-ZÁ-ÚÉÊÓÔÍ][A-ZÁ-ÚÉÊÓÔÍa-zá-úéêóôí]+\sreitor"
        prof_pattern = r"Prof\.\s[A-Z][a-z]+\s[A-Z][a-z]+"
        reitor_pattern = r"[A-Z][A-ZÁ-ÚÉÊÓÔÍ][A-ZÁ-ÚÉÊÓÔÍa-zá-úéêóôí]+\s\sReitor"

        # prof_pattern = r"PROF\s+([a-z]+\s+[a-z]+)\s+PRESIDENTE"

        # reitor_pattern = r"[A-Z][A-ZÁ-ÚÉÊÓÔÍ][A-ZÁ-ÚÉÊÓÔÍa-zá-úéêóôí]+\sREITOR"
  
        text = text.replace("\n", "")

        professor_match = re.search(prof_pattern, text)
        if professor_match:
            return professor_match.group()

        # Verifique se o texto contém a assinatura de um reitor
        reitor_match = re.search(reitor_pattern, text)
        if reitor_match:
            return reitor_match.group()
        return None
    
    def classification(self, text) -> str:
        # classificar o documento dividir entre consiuni e consepe
        if "consuni" in text.lower() or "conselho universitário" in text.lower():
            return "Consuni"
        if "consepe" in text.lower() or "conselho de ensino" in text.lower():
            return "Consepe"
        return None
    
    def old_new_classification(self) -> str:
        # classificar o documento como antigo ou novo
        resolution = self.extract_resolutions(self.text)
        if resolution:
            #print(resolution)
            parts = resolution.split("/")
            if len(parts) == 2:
                year_part = parts[1]
            try:
                year = int(year_part)
            except ValueError:
                return "Não foi possível determinar a data"
            if 2004 <= year <= 2015:
                return "Antigo"
            elif year >= 2015:
                return "Novo"
            else:
                return "Documento de data desconhecida"
        else:
            return "None"

    def extract_publication_date(self) -> str:
        # extrair a data de publicação do documento
        if not self.text:
            self.extract_text()
        datas = []
        #date_pattern = r"\d{1,2} de [a-zA-Z]+ de \d{4}"
       # date_pattern = r"palmas\s\d+\s[a-zA-Z]+\s\d{4}"

        date_pattern = r"\d{1,2} de [a-zA-Z]+ de \d{4}"
        documents = self.text.split("\n")
        for document in documents:
            match = re.search(date_pattern, document.lower())
            if match:
                # verificar demais grupos

                day = match.group().split(" ")[0]
                # print(day)
                month = match.group().split(" ")[2]
                # print(month)
                year = match.group().split(" ")[4]
                # print(year)
                month_dict = {
                    "janeiro": "01",
                    "fevereiro": "02",
                    "março": "03",
                    "abril": "04",
                    "maio": "05",
                    "junho": "06",
                    "julho": "07",
                    "agosto": "08",
                    "setembro": "09",
                    "outubro": "10",
                    "novembro": "11",
                    "dezembro": "12",
                }
                month = month_dict[month.lower()]
                date = f"{day}/{month}/{year}"

                datas.append(date)
        return datas

    def combination(self) -> dict:
        # combinar todas as funções
        
        #dic_null = []

        for i, document in enumerate(self.doc_array):
            pdf_path = document.tags.get("path")
            text = self.extract_text()
            title = self.extract_title() 
            classification = self.classification(text)
            resolution = self.extract_resolutions(text) 
            dates = self.extract_publication_date() 
            dates = dates if dates else []  # Assign an empty list to dates if it is None
            date = dates[i] if i < len(dates) else None 
            #formated_content = self.text_preprocessing(text)
            
            semantics_relatioan = SemanticRelationRecommender(text=self.text, title=title, classification=classification, resolution=resolution, pdf_path=pdf_path, dates=dates)
            semantics_relation = semantics_relatioan.semantic_relations_() 
            


            if pdf_path:
                
                combination =  {
                    "title": title,
                    "pdf_path": pdf_path,
                    "content": text,
                    #"formated_content": formated_content,
                    "status": classification,
                    "resolution": resolution,
                    "signature": self.extract_signatures(text),
                    "date": date,
                    "semantic_relation": semantics_relation,
                    "counsil": self.classification(text),
                }
            print(combination['title'])
            print("---------------------------------------------------")
            print(combination['semantic_relation'])

        
        return combination
    
    