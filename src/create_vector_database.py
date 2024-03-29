from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database(
    file_path,
    pinecone_api_key,
    openai_api_key,
    index_name,
    dimension,
    metric,
    cloud,
    region,
    model,
    embed_batch_size,
):
    """
    Creates a vector database from a given file containing data schemas.

    Parameters:
    - file_path (str): Path to the file containing the data.
    - pinecone_api_key (str): API key for Pinecone.
    - openai_api_key (str): API key for OpenAI.
    - index_name (str): Name for the Pinecone index.
    - dimension (int): Dimension for the vector index.
    - metric (str): Similarity metric for the vector index.
    - cloud (str): Cloud provider for Pinecone.
    - region (str): Region for Pinecone deployment.
    - model (str): Model name for OpenAI embedding.
    - embed_batch_size (int): Batch size for embedding generation.
    """

    if not pinecone_api_key or not openai_api_key:
        logging.error("API keys for Pinecone and OpenAI are required.")
        raise ValueError("API keys for Pinecone and OpenAI are required.")

    pc = Pinecone(api_key=pinecone_api_key)
    index_names = pc.list_indexes().names()

    if index_name not in index_names:
        pc_index = pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    else:
        logging.info("Vector Index Already Exists")
        pc_index = pc.Index(name=index_name)

    with open(
        file_path,
        "r",
    ) as f:
        text = f.read()
    schemas = text.split("&")
    for i in range(len(schemas)):
        print(schemas[i].split(" ")[2])

    docs = []
    for i, schema in enumerate(schemas):
        docs.append(
            Document(text=schema, doc_id=i, extra_info={"title": schema.split(" ")[2]})
        )

    # Set Up Embedding and Index
    vector_store = PineconeVectorStore(
        pinecone_index=pc_index, index_name=index_name, api_key=pinecone_api_key
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(
        model=model, embed_batch_size=embed_batch_size, api_key=openai_api_key
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    index = GPTVectorStoreIndex.from_documents(
        docs, storage_context=storage_context, service_context=service_context
    )

    return index


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    pinecone_api_key = os.environ.get("PINECONE_SERVERLESS_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    file_path = "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/context.txt"
    index_name = "schema-index"
    metric = "cosine"
    dimension = 1536
    cloud = "aws"
    region = "us-west-2"
    embed_batch_size = 10
    model = "text-embedding-3-small"

    index = create_database(
        file_path,
        pinecone_api_key,
        openai_api_key,
        index_name,
        dimension,
        metric,
        cloud,
        region,
        model,
        embed_batch_size,
    )
