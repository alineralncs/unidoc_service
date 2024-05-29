from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session
from database.config import get_session, engine, Base
from routers.documents.documents_service import DocumentService
from routers.documents.recommedation_service import RecommendationService
import schemas
from models.user import User, TokenTable
from utils import verify_password, create_access_token, create_refresh_token
import os
from docarray import Document, DocumentArray

from models.document import Document as DocumentModel

Base.metadata.create_all(bind=engine)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)
# request: schemas.CreateDocuments, db: Session = Depends(get_session)


@router.post("/createDocuments/")
def create_documents(document: schemas.CreateDocuments, db: Session = Depends(get_session)): 
    root_directory = "documents"

    results = []

    for year in range(2004, 2023):
        for folder_name in ["Consuni"]:
            folder_path = os.path.join(root_directory, folder_name, str(year))

            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    pdf_path = os.path.join(root, filename)

                    doc_array = DocumentArray()
                    doc_array.append(Document(path=pdf_path))

                    processor_service = DocumentService(doc_array)

                    if not processor_service.is_document_legible():
                        continue
                    combination = processor_service.combination()
                    if not combination:
                        continue 
                    results.append(combination)
    documents = []
    for result in results:
            
            document_exists = db.query(DocumentModel).filter_by(pdf_path=result['pdf_path']).first()
            if document_exists:
                document_exists.status = result['status']
                document_exists.resolution = result['resolution']
                document_exists.signature = result['signature']
                document_exists.date = result['date']
                # document_exists.content = result['formatted_content']
                document_exists.semantic_relation = result['semantic_relation']
                document_exists.counsil = result['counsil']

            else:
                document = DocumentModel(title=result['title'], pdf_path=result['pdf_path'], content=result['content'], status=result['status'], resolution=result['resolution'], signature=result['signature'], date=result['date'], semantic_relation=result['semantic_relation'], counsil=result['counsil'])
                
                documents.append(document)
        
    db.add_all(documents)
    db.commit()
    db.refresh(document)

    return {"message": "Document created successfully"}

@router.get("/getDocuments/")
def get_documents(db: Session = Depends(get_session)):
    query = db.query(DocumentModel).all()
    return query

@router.get("/getDocument/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_session)):
    selected_document = db.query(DocumentModel).filter_by(id=document_id).first()
    all_documents = db.query(DocumentModel).all()

    recommendation_service = RecommendationService(selected_document, all_documents)

    recommendations = recommendation_service.all_recommenders()

    return recommendations


@router.get("/getDocumentQuery/{query}")
def get_document_query(query: str, db: Session = Depends(get_session)):

    all_documents = db.query(DocumentModel).all()

    recommendation_service = RecommendationService(query, all_documents)

    recommendations = recommendation_service.all_recommenders()

    return recommendations


@router.get("/getDocumentQueryContent/{query}")
def get_document_content(query: str, db: Session = Depends(get_session)):
    all_documents = db.query(DocumentModel).all()

    recommendation_service = RecommendationService(query, all_documents)

    recommendations = recommendation_service.all_recommenders_content()
    return recommendations