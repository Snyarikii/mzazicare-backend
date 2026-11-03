"""SQLAlchemy column types that encrypt PHI before it reaches the database."""

import json

from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

from app.core.security_crypto import decrypt_field, encrypt_field


class EncryptedString(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return encrypt_field(value)

    def process_result_value(self, value, dialect):
        return decrypt_field(value)


class EncryptedJSON(TypeDecorator):
    """Keep the Python list/dict API while storing JSON as encrypted text."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return encrypt_field(json.dumps(value)) if value is not None else None

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # A pre-migration PostgreSQL JSON column is decoded by the driver first.
        if isinstance(value, (list, dict)):
            return value
        return json.loads(decrypt_field(value))
