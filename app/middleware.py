from fastapi import Request
from jose import JWTError
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from .utils.jwt_auth import decode_token


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = request.headers.get("Authorization")

        request.state.user = None

        if token:
            try:
                token = token.split(" ")[1]

                payload = decode_token(token)

                user_id: str | None = None
                if payload:
                    user_id = payload.get("user_id")

                if user_id is None:
                    pass

                db: Session = next(get_db())

                user = db.query(User).filter(User.id == user_id).first()

                if user:
                    request.state.user = user

            except JWTError:
                pass

        response = await call_next(request)
        return response