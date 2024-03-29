# Retrieval Augmented Generation (RAG) Model for generating SQL queries from text

This project is designed to automate the conversion of natural language text queries into SQL statements, utilizing vector databases and Large Language Models (LLMs) like OpenAI's GPT-4. It is aimed at simplifying data retrival from databases, parrticularly Electronic Health Record (EHR) systems, by translating user-friendly queries into structured SQL queries. This project is an ongoing development and might change significantly in the near future.

## Current Features
- LLM Query Handler: A Python class that interfaces with the Pinecone vector database and a OpenAI LLM to generate SQL queries from natural language input.
- CSV to SQLite Database Conversion: A script for creating SQLite databases from CSV files, making it easy to set up test databases from flat data.

## Contributions
1. Clone the repository.
2. Install the required dependencies. This project uses [Poetry](https://python-poetry.org) for dependency management.
```sh
poetry install
```
3. Set up a `.env` file with your `PINECONE_SERVERLESS_API_KEY` and `OPENAI_API_KEY`.