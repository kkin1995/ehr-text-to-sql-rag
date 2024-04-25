import streamlit as st
import sqlite3
import pandas as pd
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE = os.environ.get("VECTOR_STORE")
EMBED_MODEL = os.environ.get("EMBED_MODEL")
GPT_MODEL = os.environ.get("GPT_MODEL")
METRIC_FILENAME = os.environ.get("METRIC_FILENAME")
CONTEXT_PROMPT_FILE_PATH = os.environ.get("CONTEXT_PROMPT_FILE_PATH")


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


def display_schemas(schemas):
    st.markdown("## Semantically Similar Schemas")
    for schema in schemas:
        parts = schema.split("Schema:\n")
        description = parts[0]
        sql_statement = parts[1]
        match = re.search(r"CREATE TABLE\s+([^\s(]+)", sql_statement)

        if match:
            st.markdown(f"### {match.group(1)}")
        st.markdown(description)
        st.code(sql_statement, language="sql")


def reset_app():
    keys_to_reset = ["user_has_interacted", "context_prompt", "user_prompt"]
    for key in keys_to_reset:
        st.session_state[key] = "" if key == "user_prompt" else None


def update_query_cost(new_cost):
    data = read_metric_file()
    if data["total_cost"] is None:
        data["total_cost"] = 0.0
    data["total_cost"] += new_cost
    with open(METRIC_FILENAME, "w") as f:
        json.dump(data, f)


def get_visitor_count():
    with open(METRIC_FILENAME, "r") as f:
        data = json.load(f)
        return data["visitor_count"]


def increment_visitor_count():
    data = read_metric_file()
    data["visitor_count"] += 1
    with open(METRIC_FILENAME, "w") as f:
        json.dump(data, f)


def read_metric_file():
    if not os.path.exists(METRIC_FILENAME):
        with open(METRIC_FILENAME, "w") as f:
            data = {"total_cost": 0.0, "visitor_count": 0}
            json.dump(data, f)
        return data
    else:
        with open(METRIC_FILENAME, "r") as f:
            data = json.load(f)
        return data


try:
    with open(CONTEXT_PROMPT_FILE_PATH) as f:
        default_context_prompt = f.read()
except Exception:
    default_context_prompt = ""

increment_visitor_count()
visitor_count = get_visitor_count()
st.sidebar.write(f"Visitor Count: {visitor_count}")

user_prompt = st.text_input("User Prompt:", key="user_prompt")
st.write(f"Your prompt is: {user_prompt}")
user_has_interacted = True


ask_for_context_prompt = st.radio(
    "Do you want to enter a context prompt?",
    ["Yes", "No"],
    index=1,
    key="ask_for_context_prompt",
)
if ask_for_context_prompt == "Yes" or not default_context_prompt:
    context_prompt = st.text_input("Context prompt:", key="context_prompt")
    if not context_prompt:
        st.warning("Please enter the context prompt to proceed.")
        st.stop()
else:
    context_prompt = default_context_prompt

view_context = st.checkbox("View Context Prompt")
if view_context:
    st.write(context_prompt)

if user_prompt and user_has_interacted:
    with st.spinner("Processing..."):

        try:
            from src.query_llm import LLMQueryHandler

            handler = LLMQueryHandler(
                model=GPT_MODEL,
                vector_store=VECTOR_STORE,
                embed_model=EMBED_MODEL,
                top_k=3,
            )

            schemas = handler.get_semantic_schemas(user_prompt)
            display_schemas(schemas)

            output = handler.generate_sql_query(
                schemas,
                user_prompt,
                context=context_prompt,
            )
            sql_query = output["SQL_QUERY"]
            st.write("SQL Query:")
            st.code(sql_query, language="sql")

            cost = handler.calculate_query_execution_cost(
                GPT_MODEL, output["N_PROMPT_TOKENS"], output["N_GENERATED_TOKENS"]
            )
            update_query_cost(cost)
            data = read_metric_file()
            total_cost = data["total_cost"]
            st.write(f"Query Cost: ${cost}")
            st.write(f"Total Cost: ${total_cost}")

            df = execute_sql_on_db("patient_health_data.db", sql_query)
            st.dataframe(df)

            if st.button("Reset"):
                reset_app()
                st.experimental_rerun()
        except Exception as e:
            st.error(f"An Error Occured: {e}")
