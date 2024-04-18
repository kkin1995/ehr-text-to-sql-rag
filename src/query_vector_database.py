from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
import weaviate
from utils import check_and_get_api_keys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def query_database(
    query: str,
    vector_store: str,
    embed_model: str,
    index_name: str = None,
    top_k: int = 5,
):
    """
    Queries the vector database for items that are similar to a given query. This function
    supports querying from different vector stores, specificallyyy "pinecone" and "weaviate",
    to find the top similar items based on the similarity of their vector representations
    to the vector representation of the input query.

    Parameters:
    ----
    - query (str): The query string to search for.
    - vector_store (str): Specified the vector databse service to use for the query.
    Currently, this function only supports "pinecone" and "weaviate".
    - embed_model (str): The model identifier for the OpenAI API to be used for generating
    embeddings from the text data.
    - index_name (str, optional): The name of the index within the specified vector store
    to query against. If not provided, a default index name will be used based on the
    vector store ("schema-index for Pinecone and "SchemaIndex" for Weaviate).
    - top_k (int, optional): The number of top similar items to retrieve. It defaults to 5.

    Returns:
    ----
    - A list of nodes representing the top k similar items. Each node in the list corresponds
    to an item in the vector databse.

    Raises:
    ----
    - 'ValueError': If the 'vector_store' specified is not supported (i.e. not "pinecone"
    or "weaviate"), a 'ValueError' will be raised indicating the unsupported vector store.
    - 'ValueError': If the required API keys for the specified vector store are not provided
    or are invalid, a 'ValueError' will be raised.

    Example Usage:
    ----
    # Query the database for items similar to "machine learning"
    similar_items = query_database(
        query="machine learning",
        vector_store="pinecone",
        index_name="my-index",
        top_k=10
    )
    for node in nodes:
        print(f"Items: {node.get_text()}")
    """

    pinecone_api_key, openai_api_key = check_and_get_api_keys()

    if vector_store not in ["pinecone", "weaviate"]:
        raise ValueError(
            f"{vector_store} is not supported. Currently supported: 'pinecone' or 'weaviate'"
        )

    if vector_store == "pinecone":

        if index_name is None:
            index_name = "schema-index"

        if not pinecone_api_key or not openai_api_key:
            logger.error("Pinecone API Key and OpenAI API Key are required")
            raise ValueError("Pinecone API Key and OpenAI API Key are required")

        pc = Pinecone(api_key=pinecone_api_key)
        pc_index = pc.Index(name=index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=pc_index, api_key=pinecone_api_key
        )

    elif vector_store == "weaviate":

        if index_name is None:
            index_name = "SchemaIndex"

        client = weaviate.Client(url="http://localhost:8080")
        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name=index_name
        )

    retriever = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=OpenAIEmbedding(model=embed_model, api_key=openai_api_key),
    ).as_retriever(similarity_top_k=top_k)

    nodes = retriever.retrieve(query)

    return nodes


if __name__ == "__main__":
    query = "How does the prevalence of specific conditions (e.g., hypertension, diabetes) vary across different age groups and ethnicities within our patient population?"

    try:
        nodes = query_database(query, "weaviate", "text-embedding-3-small")
    except Exception as e:
        logger.error(f"Failed to query vector database: {e}")

    for node in nodes:
        title = node.metadata["title"]
        # print(f"Table: {title}")
        # print(f"Similarity Score: {node.get_score()}")
        print(f"Schems: {node.get_text()}")
