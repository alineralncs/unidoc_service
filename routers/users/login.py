from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session
from database.config import get_session
import schemas
from models.user import User, TokenTable
from utils import verify_password, create_access_token, create_refresh_token

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.post("/login/")
def login(request: schemas.requestDetails, db: Session = Depends(get_session)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not verify_password(request.password, user.password):
        raise HTTPException(status_code=400, detail="Invalid password")
    
    access=create_access_token(user.id)
    refresh = create_refresh_token(user.id)

    token_db = TokenTable(user_id=user.id,  acess_token=access,  refresh_token=refresh, status=True)
    db.add(token_db)
    db.commit()
    db.refresh(token_db)
    return {
        "access_token": access,
        "refresh_token": refresh,
    }