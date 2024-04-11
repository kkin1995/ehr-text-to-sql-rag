import weaviate
from utils import setup_logger

logger = setup_logger(__name__)


def delete_database(
    database_client: weaviate.Client, index_name, vector_store="weaviate"
):
    if vector_store == "weaviate":
        try:
            database_client.schema.delete_class(index_name)
            logger.info(
                f"Successfully deleted the vector index '{index_name}' from {vector_store}"
            )
        except Exception as e:
            logger.error(
                f"Error deleting the vector index '{index_name}' from {vector_store}"
            )


if __name__ == "__main__":
    WEAVIATE_URL = "http://localhost:8080"
    client = weaviate.Client(url=WEAVIATE_URL)

    vector_index_to_delete = "SchemaIndex"
    delete_database(client, vector_index_to_delete)
