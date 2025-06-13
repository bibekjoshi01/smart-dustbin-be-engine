from fastapi import HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from app.models import User


def get_current_user(request: Request) -> User:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )
    return user


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")