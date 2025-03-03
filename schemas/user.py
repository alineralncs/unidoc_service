from pydantic import BaseModel
import datetime

class CreateUser(BaseModel):
    username: str
    email: str
    password: str

class requestDetails(BaseModel):
    username: str
    password: str

class TokenSchema(BaseModel):
    access_token: str
    token_type: str

class ChangePassword(BaseModel):
    old_password: str
    new_password: str

class TokenCreate(BaseModel):
    user_id:str
    access_token:str
    refresh_token:str
    status:bool
    created_date:datetime.datetime