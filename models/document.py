from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from database.config import Base
import datetime

class Document(Base):
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    pdf_path = Column(String)
    content = Column(String)
    # formated_content = Column(String)
    created_date = Column(DateTime, default=datetime.datetime.now())
    status = Column(String)
    resolution = Column(String)
    signature = Column(String)
    date = Column(String)
    semantic_relation = Column(JSON)
    counsil = Column(String)