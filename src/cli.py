import argparse
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
from utils import get_text_hash, sanitize_filename, setup_logger
from query_llm import LLMQueryHandler

load_dotenv()

logger = setup_logger(__name__)

DB_PATH = os.environ.get("DB_PATH")
OUTPUT_DATA_PATH = os.environ.get("OUTPUT_DATA_PATH")

if not DB_PATH or not OUTPUT_DATA_PATH:
    logger.error("Enviroment Variables DB_PATH or OUTPUT_DATA_PATH not set")
    raise ValueError("Enviroment Variables DB_PATH or OUTPUT_DATA_PATH not set")

parser = argparse.ArgumentParser(
    description="This script generates and executes a SQL query based on a user prompt, vector store, and GPT model, then saves the output."
)
parser.add_argument(
    "--user_prompt",
    required=True,
    help="A user-provided text query that will be converted into a SQL query using a LLM",
)
parser.add_argument(
    "--vector_store",
    required=True,
    help="Name of the Vector DB to be used. Currently supported: pinecone or weaviate.",
)
parser.add_argument(
    "--gpt_model",
    required=True,
    help="The GPT model identifier to be used for SQL query generation.",
)

args = parser.parse_args()

user_prompt = args.user_prompt
vector_store = args.vector_store
gpt_model = args.gpt_model

logger.info(f"User Prompt: {user_prompt}")
logger.info(f"Vector Store: {vector_store}")
logger.info(f"GPT Model: {gpt_model}")


def generate_sql_query(
    user_prompt: str, vector_store: str, gpt_model: str, embed_model: str
) -> str | None:
    """
    Generates an SQL query based on the user's text prompt, vector store, and GPT model.

    Parameters:
    ----
    - user_prompt: The user's input prompt for generating the query.
    - vector_store: The datastore containing vectors for query generation.
    - gpt_model: The GPT model used for query generation.

    Returns:
    ----
    The generated SQL query as a string, or None if an error occurs.
    """
    try:
        with open(
            "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/context.txt"
        ) as f:
            context_prompt = f.read()
        handler = LLMQueryHandler(gpt_model, vector_store, embed_model, top_k=3)
        schemas = handler.get_semantic_schemas(user_prompt)
        output = handler.generate_sql_query(
            schemas, user_prompt, context=context_prompt
        )
        return output["SQL_QUERY"]
    except Exception as e:
        logger.exception(f"Error Occured in LLMQueryHandler. Exiting Program: {e}")
        exit(1)


def get_column_names_from_db(db_path: str, sql_query: str) -> list[str]:
    """
    Retrieves the column names from a SQL query execution result.

    This function connects to a SQLite database, executes a provided SQL query,
    and extracts the column names from the query result.

    Parameters:
    ----
    - db_path (str): The file path to the SQLite database.
    - sql_query (str): The SQL query to execute for which the column names are needed.

    Returns:
    ----
    - list[str]: A list of column names (str) from the SQL query result.
    """
    cols = []
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        data = cursor.execute(sql_query)
        n_cols = len(data.description)
        for i in range(n_cols):
            cols.append(data.description[i][0])

    return cols


def execute_sql_on_db(db_path: str, query: str, params=None) -> pd.DataFrame:
    """
    Executes SQL query on specified SQLite3 database with parameters and returns data in a pandas DataFrame.

    Parameters:
    ----
    - db_path (str): The file path to the SQLite database.
    - query (str): The SQL query to execute.
    - params (dict, optional): Parameters to bind to the query.

    Returns:
    ----
    - pandas.DataFrame: The result of the SQL query as a DataFrame.
    """
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(query, connection, params)


embed_model = "text-embedding-3-small"
sql_query = generate_sql_query(user_prompt, vector_store, gpt_model, embed_model)
if sql_query is None:
    logger.error("SQL Query Generation Failed. Exiting Program.")
    exit(1)
logger.info(f"Generated SQL Query: {sql_query}")

try:
    df = execute_sql_on_db(DB_PATH, sql_query)
except Exception as e:
    logger.exception(f"Error Occured While Executing SQL Query. Exiting Program: {e}")
    exit(1)

columns = get_column_names_from_db(DB_PATH, sql_query)
df.columns = columns

user_prompt_sanitized = sanitize_filename(user_prompt)
sql_query_hash = get_text_hash(sql_query)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

output_file_name = (
    f"query_result_{timestamp_str}_{user_prompt_sanitized}_{sql_query_hash}.csv"
)
os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
output_file_path = os.path.join(OUTPUT_DATA_PATH, output_file_name)

df.to_csv(output_file_path, index=False)

logger.info(f"Data saved to {output_file_name}")
