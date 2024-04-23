from query_vector_database import query_database
import pandas as pd


def write_user_query_and_semantic_schema_to_file(
    user_query: str, vector_store: str, embed_model: str, top_k: int = 5
) -> pd.DataFrame:
    """
    Queries a vector database for semantic schemas related to the user's query,
    and compiles the results into a DataFrame.

    Parameters:
    ----
    user_query (str): The query string provided by the user.
    vector_store (str): Specified the vector databse service to use for the query. Currently, this function only supports "pinecone" and "weaviate".
    embed_model (str): The model identifier for the OpenAI API to be used for generating embeddings from the text data.
    top_k (int, optional): The number of top similar items to retrieve. It defaults to 5.

    Returns:
    ----
    pd.DataFrame: A DataFrame containing the user query, related titles, similarity scores, and semantic schemas.
    """
    items = []
    nodes = query_database(user_query, vector_store, embed_model, top_k=top_k)
    for node in nodes:
        items.append(
            {
                "UserQuery": user_query,
                "Title": node.metadata["title"],
                "SimilarityScore": node.get_score(),
                "Schema": node.get_text(),
            }
        )

    return pd.DataFrame(items)


def compile_queries_to_csv(
    user_queries: list[str],
    output_file_path: str,
    vector_store: str,
    embed_model: str,
    top_k: int,
):
    """
    Compiles multiple user queries into a single CSV file containing semantic schemas and other relevant data.

    Parameters:
    ----------
    user_queries (list[str]): List of user queries to process.
    output_file_path (str, optional): The path to save the output CSV file.
    """
    dfs = [
        write_user_query_and_semantic_schema_to_file(
            query, vector_store, embed_model, top_k
        )
        for query in user_queries
    ]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_file_path)


if __name__ == "__main__":
    vector_store = "weaviate"
    embed_model = "text-embedding-3-small"
    output_file_path = "query_schema_semantic_score.csv"
    user_queries = [
        "How many male patients have diabetes alongwith hypertension?",
        "How does the prevalence of specific conditions vary across different age groups and ethnicities within our patient population?",
        "Can you list all past and current medical conditions for a given patient,including dates of diagnosis and resolution, if applicable?",
        "For patients with a specific condition (e.g.,asthma), what care plans have been most effective in managing their symptoms, based on patient outcomes and condition resolution rates?",
        "What are the most commonly prescribed medications for heart disease among our patients, and what has been the average cost for these medications?",
    ]
    top_k = 5

    compile_queries_to_csv(
        user_queries, output_file_path, vector_store, embed_model, top_k
    )
