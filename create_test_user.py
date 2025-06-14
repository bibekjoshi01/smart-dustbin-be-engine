from sqlalchemy.orm import Session
from app.database import engine
from app.models import User
from datetime import datetime

from app.utils.password import hash_password


def create_test_user(
    email: str = "admin@gmail.com", password: str = "admin", is_superuser=False
):
    db = Session(bind=engine)
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        print(f"User {email} already exists.")
        return

    user = User(
        email=email,
        password=hash_password(password),  # ✅ bcrypt hashed password
        created_at=datetime.now(),
        is_superuser=is_superuser,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    print(f"Created user: {email}")


create_test_user()
