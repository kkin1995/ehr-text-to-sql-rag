from openai import OpenAI
from query_vector_database import query_database
import logging

logging.basicConfig(level=logging.INFO)


class LLMQueryHandler:
    """
    A handler class for querying vector databases and generating SQL queries from natural language prompts using a Large Language Model (LLM).
    It essntially creates a RAG (Retrieval Augmented Generation) algorithm.

    Parameters:
    ----
    pinecone_api_key (str): API key for accessing the Pinecone vector database.
    openai_api_key (str): API key for accessing OpenAI's models.
    model (str): The name of the model to use for generating SQL queries.
    index_name (str): The name of the index within the Pinecone vector database.
    """

    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        model: str,
        index_name: str = "schema-index",
    ):
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.model = model
        self.index_name = index_name
        self.client = OpenAI(api_key=self.openai_api_key)

    def get_semantic_schemas(self, user_prompt: str) -> list[str]:
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
            pinecone_api_key=self.pinecone_api_key,
            openai_api_key=self.openai_api_key,
        )
        return [node.get_text() for node in nodes]

    def generate_sql_query(
        self, schemas: list[str], user_prompt: str, system_prompt: str = None
    ) -> dict:
        """
        Generates an SQL query from a list of semantic schemas and a user prompt using the specified LLM model.

        Parameters:
        ----
        schemas (list[str]): The list of semantic schemas related to the user's query.
        user_prompt (str): The user's query in natural language.
        system_prompt (str): The prompt used in the "system" of the LLM.

        Returns:
        ----
        output (dict): Python dictionary containing SQL_QUERY, MODEL, N_PROMPT_TOKENS, N_GENERATED_TOKENS.
        """
        if system_prompt == None:
            system_prompt = self._create_system_prompt(schemas)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        sql_query = completion.choices[0].message.content
        n_generated_tokens = completion.usage.completion_tokens
        n_prompt_tokens = completion.usage.prompt_tokens
        gpt_model = completion.model
        output = {
            "SQL_QUERY": sql_query,
            "MODEL": gpt_model,
            "N_PROMPT_TOKENS": n_prompt_tokens,
            "N_GENERATED_TOKENS": n_generated_tokens,
        }
        return output

    @staticmethod
    def _create_system_prompt(schemas: list[str]) -> str:
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

    def calculate_query_execution_cost(
        self, model: str, n_prompt_tokens: int, n_generated_tokens: int
    ) -> float:
        """
        Calculates the cost of querying with a Large Language Model (LLM) based on the model used, number of prompt tokens, and number of generated tokens.

        This method considers different pricing schemes for various models, including both GPT-4 and GPT-3.5 variants, calculating the total cost by adding the input (prompt) and output (generated) costs according to the specific rates for each model type.

        Parameters:
        ----
        model (str): The identifier of the LLM model used for the query, which determines the pricing scheme.
        n_prompt_tokens (int): The number of tokens in the prompt sent to the model.
        n_generated_tokens (int): The number of tokens generated by the model in response to the prompt.

        Returns:
        ----
        float: The total cost of the LLM query, calculated as the sum of input and output costs, in dollars.

        Notes:
        ----
        If the model is not supported, a warning is logged, and the function attempts to return a calculated cost, which may default to 0 if the model does not match any supported models. It is recommended to check for supported models before calling this method.
        """
        input_cost, output_cost = 0, 0
        if model == "gpt-4-0125-preview" or model == "gpt-4-1106-preview":
            input_cost += (n_prompt_tokens * 10) / 1e6
            output_cost += (n_generated_tokens * 30) / 1e6
        elif model == "gpt-4":
            input_cost += (n_prompt_tokens * 30) / 1e6
            output_cost += (n_generated_tokens * 60) / 1e6
        elif model == "gpt-4-32k":
            input_cost += (n_prompt_tokens * 60) / 1e6
            output_cost += (n_generated_tokens * 120) / 1e6
        elif model == "gpt-3.5-turbo-0125":
            input_cost += (n_prompt_tokens * 0.50) / 1e6
            output_cost += (n_generated_tokens * 1.50) / 1e6
        elif model == "gpt-3.5-turbo-instruct":
            input_cost += (n_prompt_tokens * 1.50) / 1e6
            output_cost += (n_generated_tokens * 2.00) / 1e6
        else:
            logging.warning(
                """
                Model not yet supported. Currently supported:
                gpt-4-0125-preview, gpt-4-1106-preview, gpt-4, gpt-4-32k, gpt-3.5-turbo-0125, gpt-3.5-turbo-instruct
                """
            )
        return input_cost + output_cost


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    pinecone_api_key = os.environ.get("PINECONE_SERVERLESS_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    user_prompt = """
    How does the prevalence of specific conditions vary across different age groups and ethnicities within our patient population?
    """
    gpt_model = "gpt-4-1106-preview"
    # gpt_model = "gpt-4-0125-preview"
    handler = LLMQueryHandler(pinecone_api_key, openai_api_key, model=gpt_model)
    schemas = handler.get_semantic_schemas(user_prompt)
    output = handler.generate_sql_query(schemas, user_prompt)

    cost = handler.calculate_query_execution_cost(
        gpt_model, output["N_PROMPT_TOKENS"], output["N_GENERATED_TOKENS"]
    )

    print(output)
    print(f"Cost = ${cost}")
