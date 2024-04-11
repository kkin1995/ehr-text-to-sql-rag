# Retrieval Augmented Generation (RAG) Model for Generating SQL Queries from Text

This project is designed to automate the conversion of natural language text queries into SQL statements, utilizing vector databases and Large Language Models (LLMs) such as OpenAI's GPT-4. Specifically designed to facilitate easier and more intuitive data retrieval from Electronic Health Record (EHR) systems, this solution aims to bridge the gap between complex database schemas and end-user query intentions. ur goal is to streamline the process of extracting meaningful insights from EHRs without requiring users to have in-depth SQL knowledge. As an evolving solution, we anticipate significant enhancements and updates to our approach.

## How It Works
1. **Query Processing**: The system accepts natural language queries from users, which are then converted into vector embeddings using one of OpenAI's embedding models.
2. **Vector Database Search**: Utilizing vector embeddings, the system searches a specified vector database to find the most relevant EHR database schemas based on the query.
3. **SQL Query Generation**: Using the most relevant schemas with descriptions as the system input, and the user's text query as user input, the LLM  generates an optimized SQL query tailored to the SQLite3 database.
4. **Data Retrieval**: The generated SQL query is executed against the EHR database, and the retrieved data is returned to the user, completing the query-to-information cycle.

## Getting Started
1. **Clone the repository** to get started with your local copy.
2. **Install dependencies**: This project uses [Poetry](https://python-poetry.org) for easy dependency management. Run the following command to install required packages:
```sh
poetry install
```
1. Set up a `.env` file with the following variables: 
   1. `PINECONE_SERVERLESS_API_KEY`: Your Pinecone serverless API Key.
   2. `OPENAI_API_KEY`: Your OpenAI API Key.
   3. `LOG_DIR`: The directory where log files are stored.
   4. `DB_PATH`: The path to the SQLite3 .db database file.
   5. `OUTPUT_DATA_DIR`: The directory where the output from the database is stored.
2. Use the `cli.py` file to get an initial idea of how the project works. You may run the file using the command:
```sh
python3 cli.py --user_prompt <user_prompt> --vector_store <vector_store> --gpt_model <gpt_model>
```
Fill in the variables given between the <> brackets.