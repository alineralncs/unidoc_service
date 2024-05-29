from pydantic import BaseModel
import datetime

class CreateDocuments(BaseModel):
    title: str
    pdf_path: str
    content: str
    # formated_content: str
    status: str
    resolution: str
    signature: str
    date: str
    semantic_relation: dict
    counsil: str