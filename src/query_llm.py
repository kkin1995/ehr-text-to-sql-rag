from openai import OpenAI
from query_vector_database import query_database


class LLMQueryHandler:
    """
    A handler class for queryin vector databases and generating SQL queries from natural language prompts using a Large Language Model (LLM).
    It essntially creates a RAG (Retrieval Augmented Generation) algorithm.

    Parameters:
    ----
    pinecone_api_key (str): API key for accessing the Pinecone vector database.
    openai_api_key (str): API key for accessing OpenAI's models.
    model (str): The name of the model to use for generating SQL queries.
    index_name (str): The name of the index within the Pinecone vector database.
    """

    def __init__(
        self, pinecone_api_key, openai_api_key, model, index_name="schema-index"
    ):
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.model = model
        self.index_name = index_name
        self.client = OpenAI(api_key=openai_api_key)

    def get_semantic_schemas(self, user_prompt):
        """
        Queries the vector database to retrieve relevant schemas based on a natural language prompt.

        Parameters:
        ---
        user_prompt (str): The user's query in natural language.

        Returns:
        ----
        list[str]: A list of texts representing the semantic schemas related to the user's query.
        """
        nodes = query_database(
            query=user_prompt,
            index_name="schema-index",
            pinecone_api_key=pinecone_api_key,
            openai_api_key=openai_api_key,
        )
        return [node.get_text() for node in nodes]

    def generate_sql_query(self, schemas, user_prompt):
        """
        Generates an SQL query from a list of semantic schemas and a user prompt using the specified LLM model.

        Parameters:
        ----
        schemas (list[str]): The list of semantic schemas related to the user's query.
        user_prompt (str): The user's query in natural language.

        Returns:
        ----
        str: The generated SQL query.
        """
        system_prompt = self._create_system_prompt(schemas)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message

    @staticmethod
    def _create_system_prompt(schemas):
        """
        Creates a detailed system prompt for the LLM based on the provided schemas.

        Parameters:
        ----
        schemas (list[str]): The list of semantic schemas to include in the prompt.

        Returns:
        ----
        str: A detailed system prompt including instructions for generating SQL queries.
        """
        return f"""
        Your task is to convert a doctor's natural language query into a specific SQL query, leveraging the schema of an Electronic Health Record (EHR) system's database.
        This process involves understanding the intent and the details of the doctor's request and accurately mapping it to a SQL query that interacts with the EHR database schema to retrive the desired information.
        Follow these steps:

        1. Understand the EHR Database Schema: Begin with a thorough analysis of the provided EHR system's SQL database schema. Pay close attention to the tables, columns, relationships, and data types to understandd how the database stores and relates medical data.
        It is important to note that some required information, such as the age of patients, may not be directly stored in the database and may need to be derived from available data.
        2. Translate Natural Language to SQL: Convert the doctor's natural languagge query into a SQL query. This step requires interpreting medical terminology and query intent. You must construct a SQL statement that accurately targets the relevant data within the EHR system, using only columnns and relationships
        defined in the schema. If you need to calculate any data, you must calculate it using the appropiate SQL functions and you must only use the available columns.
        3. Ensure SQL Query Accuracy and Efficiency: The SQL query you generate should not only accurately reflect the doctor's request but also be optimized for effiecient execution. Consider the best practices for query performance, such as selecting only necessary columns and using appropriate JOINs, and calculatingg derived values correctly.
        4. Prepare for Iterative Refinement: Anticipate the need for adjustments. Based on feedback or further clarifications, be ready to refine your SQL query to better match the doctor's informaion needs or to accomodate any additional insights about the database schema. Especially focus on correcting any assumptions about direct versus derived data.

        Output: In you output, you should only give the SQL query. Do not give any more information or text in addition to the SQL query.

        SQL Schema:

        {schemas}
        """


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    pinecone_api_key = os.environ.get("PINECONE_SERVERLESS_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    user_prompt = """
    How does the prevalence of specific conditions vary across different age groups and ethnicities within our patient population?
    """

    handler = LLMQueryHandler(pinecone_api_key, openai_api_key, model="gpt-3.5-turbo")
    schemas = handler.get_semantic_schemas(user_prompt)
    sql_query = handler.generate_sql_query(schemas, user_prompt)

    print(sql_query)
