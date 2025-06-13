from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.utils.password import check_password
from app.utils.jwt_auth import create_access_token
from app.database import get_db
from app.dependencies import oauth2_scheme
from app.models import User
from app.schemas.user import (
    UserLogin,
    UserLoginSuccess,
    UserProfileResponse,
)


router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=UserLoginSuccess)
async def user_login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email"
        )

    if not check_password(payload.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid password"
        )

    token = create_access_token(data={"sub": user.email, "user_id": user.id})

    return UserLoginSuccess(message="Login successfull", access_token=token)


@router.get("/profile", response_model=UserProfileResponse)
async def user_profile(
    request: Request,
    token: str = Depends(oauth2_scheme),
):
    """Include Bearer Token in Authorization Headers"""

    user = request.state.user

    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if not user.is_active:
        raise HTTPException(
            status_code=403, detail="Account deactivated, contact admin"
        )

    return UserProfileResponse(
        id=user.id,
        email=user.email,
        created_at=user.created_at,
        is_active=user.is_active,
    )
