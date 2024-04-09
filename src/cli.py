import argparse
import sqlite3
import pandas as pd
from query_llm import LLMQueryHandler

parser = argparse.ArgumentParser()
parser.add_argument("--user_prompt")
parser.add_argument("--vector_store")
parser.add_argument("--gpt_model")

args = parser.parse_args()

user_prompt = args.user_prompt
vector_store = args.vector_store
gpt_model = args.gpt_model


def get_sql_query(user_prompt, vector_store, gpt_model):
    handler = LLMQueryHandler(model=gpt_model, vector_store=vector_store, top_k=3)
    schemas = handler.get_semantic_schemas(user_prompt)
    output = handler.generate_sql_query(schemas, user_prompt)

    return output["SQL_QUERY"]


sql_query = get_sql_query(user_prompt, vector_store, gpt_model)

connection = sqlite3.connect("patient_health_data.db")
pd_execute = pd.read_sql_query(sql_query, connection)

cursor = connection.cursor()

data = cursor.execute(sql_query)

n_cols = len(data.description)
cols = []
for i in range(n_cols):
    cols.append(data.description[i][0])

df = pd.DataFrame(pd_execute, columns=cols)
print(df)
