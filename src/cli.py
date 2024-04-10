import argparse
import sqlite3
import pandas as pd
import logging
from dotenv import load_dotenv
import os
from datetime import datetime
from utils import DynamicPathFileHandler, get_text_hash, sanitize_filename
from query_llm import LLMQueryHandler

load_dotenv()
LOG_DIR = os.environ.get("LOG_DIR")

log_dir = LOG_DIR
log_filename = ".log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = DynamicPathFileHandler(directory=log_dir, filename=log_filename)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

parser = argparse.ArgumentParser()
parser.add_argument("--user_prompt")
parser.add_argument("--vector_store")
parser.add_argument("--gpt_model")

args = parser.parse_args()

user_prompt = args.user_prompt
vector_store = args.vector_store
gpt_model = args.gpt_model

logger.info(user_prompt)
logger.info(vector_store)
logger.info(gpt_model)


def get_sql_query(user_prompt, vector_store, gpt_model):
    handler = LLMQueryHandler(model=gpt_model, vector_store=vector_store, top_k=3)
    schemas = handler.get_semantic_schemas(user_prompt)
    output = handler.generate_sql_query(schemas, user_prompt)

    return output["SQL_QUERY"]


sql_query = get_sql_query(user_prompt, vector_store, gpt_model)

logger.info(sql_query)

connection = sqlite3.connect("patient_health_data.db")

try:
    pd_execute = pd.read_sql_query(sql_query, connection)
except Exception as e:
    logger.exception(f"Error Occured While Executing SQL Query: {e}")

cursor = connection.cursor()

data = cursor.execute(sql_query)

n_cols = len(data.description)
cols = []
for i in range(n_cols):
    cols.append(data.description[i][0])

connection.close()

df = pd.DataFrame(pd_execute, columns=cols)

output_data_dir = "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/output_data"
user_prompt_sanitized = sanitize_filename(user_prompt)
sql_query_hash = get_text_hash(sql_query)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

output_file_name = (
    f"query_result_{timestamp_str}_{user_prompt_sanitized}_{sql_query_hash}.csv"
)
output_file_path = os.path.join(output_data_dir, output_file_name)

df.to_csv(output_file_path, index=False)

logger.info(f"Data saved to {output_file_name}")
