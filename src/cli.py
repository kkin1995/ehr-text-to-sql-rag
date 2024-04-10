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
    user_prompt: str, vector_store: str, gpt_model: str
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
        handler = LLMQueryHandler(model=gpt_model, vector_store=vector_store, top_k=3)
        schemas = handler.get_semantic_schemas(user_prompt)
        output = handler.generate_sql_query(schemas, user_prompt)
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


class DatabaseHandler:
    """
    Handles database operations using a SQLite database.

    This class abstracts the database connection and query execution process,
    allowing for executing SQL queries on a SQLite database and retrieving the
    results as a pandas DataFrame.

    Attributes:
    ---
    - connection (sqlite3.Connection): The connection to the SQLite database.

    Methods:
    ---
    - execute_query(query, params=None): Executes a given SQL query and returns the result as a pandas DataFrame.
    """

    def __init__(self, db_path: str):
        """
        Initializes the DatabaseHandler with a connection to the specified SQLite database.

        Parameters:
        ---
        - db_path (str): The file path to the SQLite database.
        """
        self.connection = sqlite3.connect(db_path)

    def execute_query(self, query: str, params=None) -> pd.DataFrame:
        """
        Executes a SQL query on the database and returns the result as a pandas DataFrame.

        This method uses the pandas read_sql_query function to execute the query and
        automatically convert the results into a DataFrame. It supports parameterized queries.

        Parameters:
        ----
        - query (str): The SQL query to execute.
        - params (dict, optional): Parameters to bind to the query, for parameterized SQL execution.

        Returns:
        ----
        - pandas.DataFrame: The result of the SQL query as a DataFrame.
        """
        with self.connection:
            return pd.read_sql_query(query, self.connection, params)


sql_query = generate_sql_query(user_prompt, vector_store, gpt_model)
if sql_query is None:
    logger.error("SQL Query Generation Failed. Exiting Program.")
    exit(1)
logger.info(f"Generated SQL Query: {sql_query}")

db_handler = DatabaseHandler(DB_PATH)

try:
    df = db_handler.execute_query(sql_query)
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
