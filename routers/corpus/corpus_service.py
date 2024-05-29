import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
# ['id', 'book_id', 'best_book_id', 'work_id', 'books_count', 'isbn',
#        'isbn13', 'authors', 'original_publication_year', 'original_title',
#        'title', 'language_code', 'average_rating', 'ratings_count',
#        'work_ratings_count', 'work_text_reviews_count', 'ratings_1',
#        'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5', 'image_url',
#        'small_image_url'],
class CorpusService:
    def __init__(self, corpus_csv: str, query: str):
        self.corpus_csv = corpus_csv
        self.query = query

    def book_corpus(self):
        df = pd.read_csv(self.corpus_csv)
        return df
        
    def semantic_relations(self):
        book = self.book_corpus()

        first = book.iloc[0]
        # algorithm to extract semantic relations
        # Noun pharase extraction such as the book title and the author
        """
        exemplo:
        liosta = [
        "Titulo", "Verbo", "Objeto",

        ]
        """
        list_of_semantic_relations = []
        for index, row in book.iterrows():
            list_of_semantic_relations.append(f"{row['authors']} write {row['title']}")

        
        return list_of_semantic_relations


    def lsa_recommender(self):
        # Latent Semantic Analysis

        list_of_semantic_relations = self.semantic_relations()

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(list_of_semantic_relations)

        svd = TruncatedSVD(n_components=100)
        X_svd = svd.fit_transform(X)

        chosen_document = self.query

        chosen_document_vector = vectorizer.transform([chosen_document])
        chosen_document_vector_svd = svd.transform(chosen_document_vector)

        similarities = cosine_similarity(chosen_document_vector_svd, X_svd)

        recommendations = [relation for relation, similarity in zip(list_of_semantic_relations, similarities[0]) if similarity > 0.8] 
        similarities = similarities[0]
        recommendations_with_similarity = [(relation, similarity) for relation, similarity in zip(list_of_semantic_relations, similarities) if similarity > 0.8]
        recommendations_with_similarity.sort(key=lambda x: x[1], reverse=True)
        return recommendations_with_similarity
        # recommendations = [book for in list_of_semantic_relations, similarity in zip(list_of_semantic_relations, similarities[0]) if similarity > 0.5]
        # recommendations.sort(reverse=True)
        # return recommendations
    
