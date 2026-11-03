import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts import promote_admin


class AdminPromotionTests(unittest.TestCase):
    @patch.object(promote_admin, "SessionLocal")
    def test_promotes_existing_account_and_commits(self, session_factory):
        user = SimpleNamespace(email="admin@example.com", role="family")
        session = session_factory.return_value.__enter__.return_value
        session.execute.return_value.scalar_one_or_none.return_value = user
        output = io.StringIO()

        with redirect_stdout(output):
            result = promote_admin.promote_admin(" Admin@Example.com ")

        self.assertEqual(result, 0)
        self.assertEqual(user.role, "ADMIN")
        session.commit.assert_called_once()
        self.assertIn("[SUCCESS] User admin@example.com promoted to ADMIN role.", output.getvalue())

    @patch.object(promote_admin, "SessionLocal")
    def test_missing_account_instructs_google_login(self, session_factory):
        session = session_factory.return_value.__enter__.return_value
        session.execute.return_value.scalar_one_or_none.return_value = None
        output = io.StringIO()

        with redirect_stdout(output):
            result = promote_admin.promote_admin("missing@example.com")

        self.assertEqual(result, 1)
        session.commit.assert_not_called()
        self.assertIn("Log in once via Google OAuth first", output.getvalue())


if __name__ == "__main__":
    unittest.main()
