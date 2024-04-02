from dotenv import load_dotenv
import os

load_dotenv()


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

    return pinecone_api_key, openai_api_key
