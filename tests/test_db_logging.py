import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE = "from app.db.session import engine; print(engine.echo)"


def engine_echo(environment: str) -> str:
    """Import the session module fresh, as the server would start under that ENVIRONMENT."""
    env = {**os.environ, "ENVIRONMENT": environment}
    result = subprocess.run([sys.executable, "-c", PROBE], cwd=PROJECT_ROOT, env=env,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip().splitlines()[-1]


class SqlEchoTests(unittest.TestCase):
    def test_sql_echo_is_on_in_development(self):
        self.assertEqual(engine_echo("development"), "True")

    def test_sql_echo_is_off_outside_development(self):
        for environment in ("production", "staging", "test"):
            with self.subTest(environment=environment):
                self.assertEqual(engine_echo(environment), "False")


if __name__ == "__main__":
    unittest.main()
