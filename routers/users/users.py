from auth_bearer import JWTBearer
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.config import get_session

from models.user import User
router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)



@router.get("/users/")
def getusers( dependencies=Depends(JWTBearer()),session: Session = Depends(get_session)):
    user = session.query(User).all()
    return user