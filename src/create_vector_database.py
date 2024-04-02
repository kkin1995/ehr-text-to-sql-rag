from llama_index.core import Document
from llama_index.core.schema import TransformComponent
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone, ServerlessSpec
import weaviate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaParser(TransformComponent):
    def __call__(self, docs: list[Document], **kwargs) -> list[Document]:
        processed_docs = []
        for doc in docs:
            schemas = doc.text.split("&")
            for schema in schemas:
                title = schema.split(" ")[2]
                processed_doc = Document(text=schema, extra_info={"title": title})
                processed_docs.append(processed_doc)
        return processed_docs


def create_database(
    file_path,
    vector_store: str,
    pinecone_api_key,
    openai_api_key,
    pinecone_index_name,
    weaviate_index_name,
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
    ----
    - file_path (str): Path to the file containing the data.
    - vector_store (str): Vector Database to use.
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

    if vector_store not in ["pinecone", "weaviate"]:
        raise ValueError(
            f"{vector_store} is  not supported. Currently supported: 'pinecone' or 'weaviate'"
        )

    if vector_store == "weaviate":
        client = weaviate.Client(url="http://localhost:8080")

        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name=weaviate_index_name
        )

    elif vector_store == "pinecone":
        if not pinecone_api_key or not openai_api_key:
            logging.error("API keys for Pinecone and OpenAI are required.")
            raise ValueError("API keys for Pinecone and OpenAI are required.")

        pc = Pinecone(api_key=pinecone_api_key)
        index_names = pc.list_indexes().names()

        if pinecone_index_name not in index_names:
            pc_index = pc.create_index(
                name=pinecone_index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        else:
            logging.info("Vector Index Already Exists")
            pc_index = pc.Index(name=pinecone_index_name)

        vector_store = PineconeVectorStore(
            pinecone_index=pc_index,
            index_name=pinecone_index_name,
            api_key=pinecone_api_key,
        )

    # Data Ingestion Into Vectoe Store
    pipeline = IngestionPipeline(
        transformations=[
            SchemaParser(),
            OpenAIEmbedding(
                model=model, api_key=openai_api_key, embed_batch_size=embed_batch_size
            ),
        ],
        vector_store=vector_store,
    )

    with open(
        file_path,
        "r",
    ) as f:
        text = f.read()

    initial_docs = [Document(text=text)]
    docs = pipeline.run(documents=initial_docs)

    index = VectorStoreIndex.from_vector_store(vector_store)

    return index


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    pinecone_api_key = os.environ.get("PINECONE_SERVERLESS_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    file_path = "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/context.txt"
    pinecone_index_name = "schema-index"
    weaviate_index_name = "SchemaIndex"
    vector_store = "weaviate"
    metric = "cosine"
    dimension = 1536
    cloud = "aws"
    region = "us-west-2"
    embed_batch_size = 10
    model = "text-embedding-3-small"

    index = create_database(
        file_path,
        vector_store,
        pinecone_api_key,
        openai_api_key,
        pinecone_index_name,
        weaviate_index_name,
        dimension,
        metric,
        cloud,
        region,
        model,
        embed_batch_size,
    )
