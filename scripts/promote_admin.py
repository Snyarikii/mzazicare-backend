"""Promote an existing Google account to the administrator role.

Run from any directory: python scripts/promote_admin.py admin@example.com
"""
import argparse
import sys
from pathlib import Path

# Direct script execution puts scripts/ on sys.path, so add the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import SessionLocal
from app.models.entities import User


def promote_admin(email: str) -> int:
    normalized_email = email.strip().casefold()
    if not normalized_email:
        print("[ERROR] Provide the email address to promote.")
        return 2

    try:
        with SessionLocal() as session:
            user = session.execute(
                select(User).where(func.lower(User.email) == normalized_email)
            ).scalar_one_or_none()
            if user is None:
                print(f"[NOT FOUND] {email} has no account row. Log in once via Google OAuth first, complete registration if prompted, then rerun this command.")
                return 1
            user.role = "ADMIN"
            session.commit()
            print(f"[SUCCESS] User {user.email} promoted to ADMIN role.")
            return 0
    except SQLAlchemyError as exc:
        print(f"[ERROR] Admin promotion failed: {exc}")
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote an existing MzaziCare user to ADMIN.")
    parser.add_argument("email", help="Email address of the existing Google account")
    arguments = parser.parse_args()
    raise SystemExit(promote_admin(arguments.email))
