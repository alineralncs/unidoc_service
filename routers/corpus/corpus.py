from fastapi import APIRouter
from .corpus_service import CorpusService

router = APIRouter(
    prefix="/corpus",
    tags=["corpus"],
    responses={404: {"description": "Not found"}},
)

@router.post("/corpus/{Query}")
def corpus(Query: str):
    book_file = "documents/books/books.csv"
    corpus_service = CorpusService(book_file, Query)

    recomendations = corpus_service.lsa_recommender()

    return recomendations