from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
import weaviate
from dotenv import load_dotenv
import os

def query_database(
    query: str,
    vector_store: str,
    embed_model: str,
    embed_batch_size: int = 10,
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
    - embed_batch_size (int): The number of documents to process in each batch when generating embeddings.
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

    load_dotenv()

    if vector_store not in ["pinecone", "weaviate"]:
        raise ValueError(
            f"{vector_store} is not supported. Currently supported: 'pinecone' or 'weaviate'"
        )

    if vector_store == "pinecone":
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if pinecone_api_key is None:
            raise ValueError(
                "PINECONE_API_KEY must be specified as an environment variable."
            )
        if index_name is None:
            index_name = "schema-index"

        pc = Pinecone(api_key=pinecone_api_key)
        pc_index = pc.Index(name=index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=pc_index, api_key=pinecone_api_key
        )

    elif vector_store == "weaviate":

        if index_name is None:
            index_name = "SchemaIndex"

        WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST")
        client = weaviate.Client(url=WEAVIATE_HOST)
        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name=index_name
        )

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY must be specified as an environment variable.")

    retriever = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=OpenAIEmbedding(
            model=embed_model, embed_batch_size=embed_batch_size, api_key=openai_api_key
        ),
    ).as_retriever(similarity_top_k=top_k)

    nodes = retriever.retrieve(query)

    return nodes


if __name__ == "__main__":
    from utils import setup_logger

    logger = setup_logger(__name__)
    query = "How does the prevalence of specific conditions (e.g., hypertension, diabetes) vary across different age groups and ethnicities within our patient population?"

    try:
        nodes = query_database(query, "weaviate", "text-embedding-3-small")
    except Exception as e:
        logger.error(f"Failed to query vector database: {e}")

    for node in nodes:
        title = node.metadata["title"]
        print(f"Table: {title}")
        print(f"Similarity Score: {node.get_score()}")
        print(f"Schems: {node.get_text()}")
