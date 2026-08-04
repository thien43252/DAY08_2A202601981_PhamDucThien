import os
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.env_utils import get_api_key, load_project_env


class TestEnvLoading(unittest.TestCase):
    def test_load_project_env_reads_repo_dotenv(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPEN_AI_API_KEY", None)

        loaded = load_project_env()

        self.assertTrue(loaded)
        self.assertTrue(get_api_key("OPENAI_API_KEY", "OPEN_AI_API_KEY"))


if __name__ == "__main__":
    unittest.main()