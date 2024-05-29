from sqlalchemy import Column, Integer, String, Boolean, DateTime
from database.config import Base
import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    is_active = Column(Boolean, default=True)

class TokenTable(Base):
    __tablename__ = "token"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    acess_token = Column(String)
    refresh_token = Column(String)
    status = Column(Boolean)
    created_date = Column(DateTime, default=datetime.datetime.now())