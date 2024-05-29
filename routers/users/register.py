import schemas 
import models 
from models.user import User
from database.config import Base, engine, SessionLocal
from fastapi import APIRouter, Depends, HTTPException, status, FastAPI
from sqlalchemy.orm import Session
from utils import get_hashed_password
from database.config import get_session


Base.metadata.create_all(bind=engine)

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

app = FastAPI()


@router.post("/register/")
async def register_user(user: schemas.CreateUser, db: Session = Depends(get_session)):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username is already taken")
    encrypter_password = get_hashed_password(user.password)

    new_user = User(username=user.username, email=user.email, password=encrypter_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}