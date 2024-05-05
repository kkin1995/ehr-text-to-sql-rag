from dotenv import load_dotenv
import os
import sys
import hashlib
import re
from datetime import datetime
import logging

load_dotenv()


class DynamicPathFileHandler(logging.FileHandler):
    def __init__(
        self,
        directory,
        filename,
        mode="a",
        encoding=None,
        delay=False,
    ):
        self.base_directory = directory
        self.base_filename = filename
        filepath = self._calculate_dynamic_path()

        super().__init__(filepath, mode, encoding, delay)

    def _calculate_dynamic_path(self):
        date_now = datetime.now()
        directory = os.path.join(
            self.base_directory, str(date_now.year), date_now.strftime("%B")
        )
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(
            directory, date_now.strftime("%d%m%Y") + self.base_filename
        )
        self.currently_logging_to = filepath
        return filepath

    def emit(self, record):
        module_name = os.path.basename(
            sys.argv[0] if sys.argv[0] else "unknown_program"
        )
        record_program = os.path.splitext(module_name)[0]
        super().emit(record)


def setup_logger(name=__name__):
    LOG_DIR = os.environ.get("LOG_DIR")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = DynamicPathFileHandler(directory=LOG_DIR, filename=".log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def sanitize_filename(input_str):
    return re.sub(r"[^a-zA-Z0-9_]", "_", input_str)


def get_text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def check_valid_vector_store(vector_store_name: str) -> bool:
    if vector_store_name not in ["pinecone", "weaviate"]:
        return False
    else:
        return True


def check_and_get_api_keys() -> tuple[str, str]:
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if pinecone_api_key is None:
        raise ValueError(
            "PINECONE_API_KEY must be specified as an environment variable."
        )
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY must be specified as an environment variable.")
    claude_api_key = os.environ.get("CLAUDE_API_KEY")
    if claude_api_key is None:
        raise ValueError("CLAUDE_API_KEY must be specified as an environment variable.")

    return pinecone_api_key, openai_api_key, claude_api_key
