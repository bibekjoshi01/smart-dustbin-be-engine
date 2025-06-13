from pydantic import BaseModel, EmailStr, StringConstraints
from typing import Annotated
from datetime import datetime


class UserLogin(BaseModel):
    email: EmailStr
    password: Annotated[str, StringConstraints()]


class UserLoginSuccess(BaseModel):
    message: str
    access_token: str


class UserProfileResponse(BaseModel):
    id: int
    email: str
    created_at: datetime
    is_active: bool
