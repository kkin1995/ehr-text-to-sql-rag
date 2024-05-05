from llama_index.core import Document
from llama_index.core.schema import TransformComponent
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone, ServerlessSpec
import weaviate
import re
from dotenv import load_dotenv
import os
from utils import check_valid_vector_store

load_dotenv()

class SchemaParser(TransformComponent):
    def __call__(self, docs: list[Document], **kwargs) -> list[Document]:
        processed_docs = []
        for doc in docs:
            schemas = doc.text.split("&")
            for schema in schemas:
                matched_string = re.search(r"CREATE TABLE (\w+)", schema)
                if matched_string:
                    title = matched_string.group(1)
                processed_doc = Document(text=schema, extra_info={"title": title})
                processed_docs.append(processed_doc)
        return processed_docs


def check_weaviate_vector_store_exists(client: weaviate.Client, vector_store_name: str):
    schema = client.schema.get()
    classes = schema.get("classes", [])
    for cls in classes:
        if cls["class"] == vector_store_name:
            return True
    return False


def initialize_vector_store(
    vector_store_name: str,
    pinecone_api_key: str,
    pinecone_config: dict,
    index_name: str = None,
) -> WeaviateVectorStore | PineconeVectorStore:
    """
    Initializes and returns a vector store instance based on the specified vector store name.
    Supports the initialization of 'Pinecone' and 'Weaviate' vector stores by configuring
    them with the given parameters. For Pinecone, it also checks and validates the provided
    configuration against required settings.

    Parameters:
    ----------
    vector_store_name (str): The name of the vector store to initialize. Currently supports 'pinecone' and 'weaviate'.

    pinecone_api_key (str): The API key for Pinecone. Required if 'pinecone' is specified as the vector store.

    pinecone_config (dict): Configuration settings specific to Pinecone, including 'metric', 'dimension', 'cloud',
    and 'region'. This parameter is mandatory and must be a dictionary when 'pinecone' is the selected vector store.

    index_name (str), optional: The name of the index to be used or created within the vector store. If not provided,
    defaults to 'SchemaIndex' for Weaviate and 'schema-index' for Pinecone.

    Returns:
    -------
    vector_store (WeaviateVectorStore | PineconeVectorStore): An instance of the vector store configured with the
    provided parameters.

    Raises:
    ------
    ValueError
        If 'pinecone_config' is not specified or is None when 'pinecone' is the vector store.
        If the Pinecone API key is not provided when 'pinecone' is the vector store.
        If 'pinecone_config' does not contain all required configurations.

    TypeError
        If 'pinecone_config' is not a dictionary.

    Notes:
    -----
    - The function performs validation checks specifically for the Pinecone configuration
      to ensure all required parameters are included. It raises errors for
      missing or invalid configurations.
    - For Weaviate, the function assumes a local instance running on 'http://localhost:8080'.

    Example:
    -------
    ### Initialize a Pinecone vector store
    pinecone_config = {
        "metric": "cosine",
        "dimension": 768,
        "cloud": "aws",
        "region": "us-west-1",
    }
    vector_store = initialize_vector_store(
        vector_store_name="pinecone",
        pinecone_api_key="your_pinecone_api_key",
        pinecone_config=pinecone_config,
        index_name="your_index_name"
    )
    """

    WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST")

    if vector_store_name == "weaviate":
        if index_name is None:
            index_name = "SchemaIndex"

        client = weaviate.Client(url=WEAVIATE_HOST)

        if not check_weaviate_vector_store_exists(client, index_name):
            vector_store = WeaviateVectorStore(
                weaviate_client=client, index_name=index_name
            )
        else:
            return False
        # logger.info(f"Created {vector_store_name} vector store")

    elif vector_store_name == "pinecone":
        if pinecone_config == None:
            raise ValueError("pinecone_config must be specified if using Pinecone.")
        elif not isinstance(pinecone_config, dict):
            raise TypeError("pinecone_config must be a dictionary.")

        required_pinecone_configs = ["metric", "dimension", "cloud", "region"]
        if not all(key in pinecone_config for key in required_pinecone_configs):
            raise ValueError("pinecone_config has missing configurations")

        metric = pinecone_config["metric"]
        dimension = pinecone_config["dimension"]
        cloud = pinecone_config["cloud"]
        region = pinecone_config["region"]

        if index_name is None:
            index_name = "schema-index"

        if not pinecone_api_key:
            # logger.error("API keys for Pinecone is required.")
            raise ValueError("API keys for Pinecone is required.")

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
            # logger.info("Vector Index Already Exists")
            pc_index = pc.Index(name=index_name)

        vector_store = PineconeVectorStore(
            pinecone_index=pc_index,
            index_name=index_name,
            api_key=pinecone_api_key,
        )

    return vector_store


def create_database(
    file_path: str,
    vector_store_name: str,
    model: str,
    embed_batch_size: int,
    pinecone_config: dict = None,
    index_name: str = None,
) -> VectorStoreIndex:
    """
    Creates and populates a vector database from a specified file containing data schemas,
    utilizing either Pinecone or Weaviate as the vector store and OpenAI embeddings for
    vectorization. The process involves reading the data, generating embeddings, and
    storing these embeddings in the vector database.

    Parameters:
    ----------
    file_path (str): The file system path to the text file containing the data schemas. Each schema
    should be separated by "&" in the file.

    vector_store_name (str): The name of the vector store to use for storing the embeddings. Supported values
    are 'pinecone' and 'weaviate'.

    model (str): The model identifier for the OpenAI API to be used for generating embeddings from the text data.

    embed_batch_size (int): The number of documents to process in each batch when generating embeddings.

    pinecone_config (dict), optional: Configuration settings for Pinecone, including 'metric', 'dimension', 'cloud',
        and 'region'. Required if using Pinecone as the vector store.

    index_name (str), optional: The name of the index to create or use within the vector store. If not specified,
        defaults to 'SchemaIndex' for Weaviate and 'schema-index' for Pinecone.

    Returns:
    -------
    VectorStoreIndex
        An instance of VectorStoreIndex, which represents the created and populated vector
        index in the specified vector store.

    Raises:
    ------
    ValueError
        If the vector store name is not supported, if required API keys are not found in
        environment variables, or if the `pinecone_config` is incomplete or missing when
        required.

    Notes:
    -----
    - The function automatically loads required API keys (PINECONE_API_KEY and OPENAI_API_KEY)
      from environment variables. Ensure these are set before calling the function.
    - This function integrates several components: reading data from a file, parsing and
      processing the data, generating embeddings using OpenAI's API, and storing these
      embeddings in the specified vector store.

    Example:
    -------
    # Create a vector database using Weaviate and OpenAI's text-embedding model
    create_database(
        file_path="path/to/data.txt",
        vector_store_name="weaviate",
        model="text-embedding-ada-001",
        embed_batch_size=5
    )
    """

    if not check_valid_vector_store(vector_store_name):
        raise ValueError(
            f"{vector_store_name} is  not supported. Currently supported: 'pinecone' or 'weaviate'"
        )

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if pinecone_api_key is None:
        raise ValueError(
            "PINECONE_API_KEY must be specified as an environment variable."
        )
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY must be specified as an environment variable.")

    vector_store = initialize_vector_store(
        vector_store_name, pinecone_api_key, pinecone_config, index_name
    )
    if vector_store:
        # Data Ingestion Into Vector Store
        pipeline = IngestionPipeline(
            transformations=[
                SchemaParser(),
                OpenAIEmbedding(
                    model=model,
                    api_key=openai_api_key,
                    embed_batch_size=embed_batch_size,
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
    else:
        return None


if __name__ == "__main__":
    file_path = "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/schemas_1.txt"
    vector_store_name = "weaviate"
    pinecone_config = {
        "metric": "cosine",
        "dimension": 1536,
        "cloud": "aws",
        "region": "us-west-2",
    }
    embed_batch_size = 10
    model = "text-embedding-3-small"

    index = create_database(
        file_path, vector_store_name, model, embed_batch_size, pinecone_config
    )
