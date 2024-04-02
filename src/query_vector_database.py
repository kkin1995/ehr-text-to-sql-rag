from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from pinecone import Pinecone
import weaviate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def query_database(
    query,
    vector_store,
    pinecone_index_name,
    pinecone_api_key,
    weaviate_index_name,
    openai_api_key,
    top_k=5,
):
    """
    Queries the vector database for the top k similar items based on the query.

    Parameters:
    - query (str): The query string to search for.
    - index_name (str): The name of the Pinecone index to query.
    - top_k (int): The number of top similar items to retrieve.

    Returns:
    - A list of nodes representing the top k similar items.
    """

    if vector_store not in ["pinecone", "weaviate"]:
        raise ValueError(
            f"{vector_store} is  not supported. Currently supported: 'pinecone' or 'weaviate'"
        )

    if vector_store == "pinecone":
        if not pinecone_api_key or not openai_api_key:
            logger.error("Pinecone API Key and OpenAI API Key are required")
            raise ValueError("Pinecone API Key and OpenAI API Key are required")

        pc = Pinecone(api_key=pinecone_api_key)
        pc_index = pc.Index(name=pinecone_index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=pc_index, api_key=pinecone_api_key
        )

    elif vector_store == "weaviate":
        client = weaviate.Client(url="http://localhost:8080")
        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name=weaviate_index_name
        )

    retriever = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    ).as_retriever(similarity_top_k=top_k)

    nodes = retriever.retrieve(query)

    return nodes


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    pinecone_api_key = os.environ.get("PINECONE_SERVERLESS_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    pinecone_index_name = "schema-index"
    weaviate_index_name = "SchemaIndex"

    query = "How does the prevalence of specific conditions (e.g., hypertension, diabetes) vary across different age groups and ethnicities within our patient population?"

    try:
        nodes = query_database(
            query,
            "weaviate",
            pinecone_index_name,
            pinecone_api_key,
            weaviate_index_name,
            openai_api_key,
        )
    except Exception as e:
        logger.error(f"Failed to query vector database: {e}")

    for node in nodes:
        title = node.metadata["title"]
        # print(f"Table: {title}")
        # print(f"Similarity Score: {node.get_score()}")
        print(f"Schems: {node.get_text()}")
