from openai import OpenAI
from query_vector_database import query_database
import logging
from utils import check_and_get_api_keys

logging.basicConfig(level=logging.INFO)


class LLMQueryHandler:
    """
    A handler class for querying vector databases and generating SQL queries from natural language prompts using a Large Language Model (LLM).
    It essntially creates a RAG (Retrieval Augmented Generation) algorithm.

    Parameters:
    ----
    model (str): The name of the model to use for generating SQL queries.
    index_name (str, optional): The name of the index within the vector database.
    """

    def __init__(self, model: str, vector_store: str, index_name: str = None, top_k=5):
        self.pinecone_api_key, self.openai_api_key = check_and_get_api_keys()
        self.model = model
        self.vector_store = vector_store
        self.index_name = index_name
        self.top_k = top_k
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
            vector_store=self.vector_store,
            index_name=self.index_name,
            top_k=self.top_k,
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
        SQL Schema:

        {schemas}

        As a expert data scientist specialising in the medical field, your task is to convert 
        a doctor’s natural language query into a specific SQL query that will be run on a 
        SQLite3 database to fetch data to answer the doctor’s query. You have to use the schema 
        of an Electronic Health Record (EHR) system’s database. Your role requires a deep understanding 
        of the database schema, and the ability to accurately interpret medical terminology and query 
        intent.

        Begin by thoroughly analysing the provided schema of the EHR system. The schema given in this message has been 
        pre-determined to be the most important for the doctor’s query you are about to analyse. 
        Identify the key attributes requested by the doctor in the query. Be mindful of attributes 
        requested by the doctor which are not directly available in the database. For example, the 
        patient’s age is not directly available and needs to be calculated from the patient’s birth 
        dates.

        The SQL query you generate should not only accurately reflect the doctor's request but also 
        be optimized for efficient execution. Consider the best practices for query performance, 
        such as selecting only necessary columns and using appropriate JOINs, and calculating 
        derived values correctly. The SQL query should conform to the SQLite3 standard.

        In any SQL statement that uses WHERE clauses, do not use the equals (=) operator, instead use the LIKE
        command with the wildcard operator.

        Output:

        Your output should consist solely of the SQL query ready to be executed on the SQLite3 database. 
        Do not give any more information or text in addition to the SQL query.
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
    # user_prompt = """
    # How does the prevalence of specific conditions vary across different age groups and ethnicities within our patient population?
    # """
    user_prompt = "Can you list all past and current medical conditions for a given patient, including dates of diagnosis and resolution, if applicable?"
    vector_store = "weaviate"
    gpt_model = "gpt-3.5-turbo-0125"
    # gpt_model = "gpt-4-0125-preview"
    handler = LLMQueryHandler(model=gpt_model, vector_store=vector_store, top_k=3)
    schemas = handler.get_semantic_schemas(user_prompt)
    output = handler.generate_sql_query(schemas, user_prompt)

    cost = handler.calculate_query_execution_cost(
        gpt_model, output["N_PROMPT_TOKENS"], output["N_GENERATED_TOKENS"]
    )

    print(f"SQL Query: \n\n {output['SQL_QUERY']}")
    print("--------------------------------")
    for idx, schema in enumerate(schemas):
        print(idx)
        print("--------------------------------")
        print(schema)
    print("--------------------------------")
    print(f"Cost = ${cost:.5f}")
    print(f"Model: {output['MODEL']}")
    print(f"Number of Prompt Tokens: {output['N_PROMPT_TOKENS']}")
    print(f"Number of Generated Tokens: {output['N_GENERATED_TOKENS']}")
