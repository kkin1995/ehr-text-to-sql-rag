from openai import OpenAI
import anthropic
import pandas as pd
import sqlite3
import re
import os
from query_vector_database import query_database

class LLMQueryHandler:
    """
    A handler class for querying vector databases and generating SQL queries from natural language prompts using a Large Language Model (LLM).
    It essntially creates a RAG (Retrieval Augmented Generation) algorithm.

    Parameters:
    ----
    model (str): The name of the model to use for generating SQL queries. Refer "Supported models".
    vector_store (str): The name of the vector store to query for augmenting the context prompt.
    embed_model (str): The name of the embedding model to use for the vector store.
    index_name (str, optional): The name of the index within the vector database.
    top_k (int, optional): Number of most similar entries to return from vector store.

    Notes:
    ----
    Supported models:
    1. OpenAI: ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct"]
    2. Claude: ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    """

    def __init__(
        self,
        model: str,
        vector_store: str,
        embed_model: str,
        db_path: str,
        index_name: str = None,
        top_k=5,
    ):
        self.model = model
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.db_path = db_path
        self.index_name = index_name
        self.top_k = top_k

        self.messages = []

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
            embed_model=self.embed_model,
            index_name=self.index_name,
            top_k=self.top_k,
        )
        return [node.get_text() for node in nodes]

    def generate_initial_query(
        self,
        schemas: list[str],
        user_prompt: str,
        context: str,
        system_prompt: str = None,
    ):
        if system_prompt == None:
            logger.info("System Prompt Not Given. Creating System Prompt...")
            system_prompt = self._create_system_prompt(schemas, context)

        model_service = self._find_model()
        logger.info(f"Using Model From: {model_service}")
        if model_service == "gpt":
            logger.info(("Using GPT"))
            self.messages.append({"role": "system", "content": system_prompt})
            self.messages.append({"role": "user", "content": user_prompt})
            logger.info(
                f"Inserted System Prompt and First User Prompt into Messages: {self.messages}"
            )
        elif model_service == "claude":
            logger.info("Using Claude")
            self.messages.append({"role": "user", "content": user_prompt})
            logger.info(
                f"Inserted System Prompt and First User Prompt into Messages: {self.messages}"
            )

    def generate_sql_query(
        self,
    ) -> dict:
        """
        Generates an SQL query from a list of semantic schemas and a user prompt using the specified LLM model.

        Parameters:
        ----
        schemas (list[str]): The list of semantic schemas related to the user's query.
        user_prompt (str): The user's query in natural language.
        context (str): The context prompt to be concatenated with the schemas.
        system_prompt (str): The prompt used in the "system" of the LLM.

        Returns:
        ----
        output (dict): Python dictionary containing SQL_QUERY, MODEL, N_PROMPT_TOKENS, N_GENERATED_TOKENS.
        """

        if self._find_model() == "gpt":
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError(
                    "OPENAI_API_KEY must be specified as an environment variable."
                )

            self.client = OpenAI(api_key=openai_api_key)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
            )
            sql_query = completion.choices[0].message.content
            n_generated_tokens = completion.usage.completion_tokens
            n_prompt_tokens = completion.usage.prompt_tokens
            model = completion.model
            output = {
                "SQL_QUERY": sql_query,
                "MODEL": model,
                "N_PROMPT_TOKENS": n_prompt_tokens,
                "N_GENERATED_TOKENS": n_generated_tokens,
            }
            return output
        elif self._find_model() == "claude":
            claude_api_key = os.environ.get("CLAUDE_API_KEY")
            if claude_api_key is None:
                raise ValueError(
                    "CLAUDE_API_KEY must be specified as an environment variable."
                )
            client = anthropic.Anthropic(api_key=claude_api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=self.system_prompt,
                messages=self.messages,
            )
            sql_query = message.content[0].text
            model = message.model
            n_prompt_tokens = message.usage.input_tokens
            n_generated_tokens = message.usage.output_tokens
            output = {
                "SQL_QUERY": sql_query,
                "MODEL": model,
                "N_PROMPT_TOKENS": n_prompt_tokens,
                "N_GENERATED_TOKENS": n_generated_tokens,
            }
            return output

    def _find_model(self):
        pattern = r"(gpt|claude)"
        match = re.search(pattern, self.model)
        if match:
            return match.group()
        else:
            return None

    def _find_claude_model(self):
        if self._find_model() == "claude":
            pattern = r"(opus|sonnet|haiku)"
            match = re.search(pattern, self.model)
            if match:
                return match.group()
            else:
                return None
        else:
            return None

    def _create_system_prompt(self, schemas: list[str], context: str) -> str:
        """
        Creates a detailed system prompt for the LLM based on the provided schemas.

        Parameters:
        ----
        schemas (list[str]): The list of semantic schemas to include in the prompt.

        Returns:
        ----
        str: A detailed system prompt including instructions for generating SQL queries.
        """
        self.system_prompt = f"""
        SQL Schema:

        {schemas}

        {context}
        """

    def process_user_query(
        self,
        schemas,
        user_prompt,
        context,
        system_prompt=None,
        retry_count=0,
        max_retries=3,
    ):
        if retry_count > max_retries:
            return None, None

        self.generate_initial_query(schemas, user_prompt, context, system_prompt)
        logger.info("Generating SQL Query from LLM")
        output = self.generate_sql_query()
        sql_query = output["SQL_QUERY"]
        logger.info(f"SQL Query generated on First Try: {sql_query}")
        df, error = self.execute_sql_on_db(self.db_path, sql_query)
        cost = self.calculate_query_execution_cost(
            output["N_PROMPT_TOKENS"], output["N_GENERATED_TOKENS"]
        )
        logger.info(f"Cost = ${cost:.5f}")
        logger.info(f"Number of Prompt Tokens: {output['N_PROMPT_TOKENS']}")
        logger.info(f"Number of Generated Tokens: {output['N_GENERATED_TOKENS']}")
        if error:
            logger.info(f"Re-prompting the LLM due to Error: {error}")
            user_reprompt = f"""
                The SQL query generated from your request resulted in an error when executed against the database. 
                Here's the error message provided by the database:

                {error}

                Please review the natural language text query again to address the issues described above and avoid technical terms or 
                database-specific jargon that might have caused the error. Here's the original query for your reference:

                {user_prompt}

                Adjust the query so it conforms to the database schema.

            """
            logger.info(f"Re-prompting the LLM with {user_reprompt}")
            self.messages.extend(
                [
                    {"role": "assistant", "content": sql_query},
                    {"role": "user", "content": user_reprompt},
                ],
            )
            output = self.generate_sql_query()
            sql_query = output["SQL_QUERY"]
            df, _ = self.execute_sql_on_db(self.db_path, sql_query)
            cost = self.calculate_query_execution_cost(
                output["N_PROMPT_TOKENS"], output["N_GENERATED_TOKENS"]
            )
            logger.info(f"Cost = ${cost:.5f}")
            logger.info(f"Number of Prompt Tokens: {output['N_PROMPT_TOKENS']}")
            logger.info(f"Number of Generated Tokens: {output['N_GENERATED_TOKENS']}")
            return df, output
        elif df.empty:
            logger.info("Re-prompting the LLM due to empty table output")
            user_reprompt = f"""

                The SQL query executed successfully but returned no results. This could happen for several reasons, 
                such as filtering criteria being too restrictive or querying data that doesn't exist.

                Please review the natural language text query and consider adjusting it to broaden the search criteria or 
                correct any inaccuracies. Here's your original prompt for reference:

                {user_prompt}

                Additionally, ensure the query aligns with the available data as described in the database schemas below:

                {schemas}

            """
            logger.info(f"Re-prompting the LLM with {user_reprompt}")
            self.messages.extend(
                [
                    {"role": "assistant", "content": sql_query},
                    {"role": "user", "content": user_reprompt},
                ],
            )
            output = self.generate_sql_query()
            sql_query = output["SQL_QUERY"]
            df, _ = self.execute_sql_on_db(self.db_path, sql_query)
            cost = self.calculate_query_execution_cost(
                output["N_PROMPT_TOKENS"], output["N_GENERATED_TOKENS"]
            )
            logger.info(f"Cost = ${cost:.5f}")
            logger.info(f"Number of Prompt Tokens: {output['N_PROMPT_TOKENS']}")
            logger.info(f"Number of Generated Tokens: {output['N_GENERATED_TOKENS']}")
            return df, output
        else:
            return df, output

    def execute_sql_on_db(
        self, db_path: str, query: str, params=None
    ) -> tuple[pd.DataFrame | None, None | str]:
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
        try:
            with sqlite3.connect(db_path) as connection:
                df = pd.read_sql_query(query, connection, params)
                return df, None
        except Exception as e:
            return None, str(e)

    def calculate_query_execution_cost(
        self, n_prompt_tokens: int, n_generated_tokens: int
    ) -> float:
        """
        Calculates the cost of querying with a Large Language Model (LLM) based on the model used,
        number of prompt tokens, and number of generated tokens.

        This method considers different pricing schemes for various models, including both GPT-4 and GPT-3.5
        variants, calculating the total cost by adding the input (prompt) and output (generated) costs
        according to the specific rates for each model type.

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
        If the model is not supported, a warning is logged, and the function attempts to return a
        calculated cost, which may default to 0 if the model does not match any supported models.
        It is recommended to check for supported models before calling this method.

        Supported models:
        1. OpenAI: ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct"]
        2. Claude: ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        """
        input_cost, output_cost = 0, 0
        if self.model == "gpt-4-0125-preview" or self.model == "gpt-4-1106-preview":
            input_cost += (n_prompt_tokens * 10) / 1e6
            output_cost += (n_generated_tokens * 30) / 1e6
        elif self.model == "gpt-4":
            input_cost += (n_prompt_tokens * 30) / 1e6
            output_cost += (n_generated_tokens * 60) / 1e6
        elif self.model == "gpt-4-32k":
            input_cost += (n_prompt_tokens * 60) / 1e6
            output_cost += (n_generated_tokens * 120) / 1e6
        elif self.model == "gpt-3.5-turbo-0125":
            input_cost += (n_prompt_tokens * 0.50) / 1e6
            output_cost += (n_generated_tokens * 1.50) / 1e6
        elif self.model == "gpt-3.5-turbo-instruct":
            input_cost += (n_prompt_tokens * 1.50) / 1e6
            output_cost += (n_generated_tokens * 2.00) / 1e6
        elif self._find_claude_model() == "opus":
            input_cost = (n_prompt_tokens * 15) / 1e6
            output_cost = (n_generated_tokens * 75) / 1e6
        elif self._find_claude_model() == "sonnet":
            input_cost = (n_prompt_tokens * 3) / 1e6
            output_cost = (n_generated_tokens * 15) / 1e6
        elif self._find_claude_model() == "haiku":
            input_cost = (n_prompt_tokens * 0.25) / 1e6
            output_cost = (n_generated_tokens * 1.25) / 1e6
        # else:
        #     logger.warning(
        #         """
        #         Model not yet supported. Currently supported:
        #         gpt-4-0125-preview, gpt-4-1106-preview, gpt-4, gpt-4-32k, gpt-3.5-turbo-0125, gpt-3.5-turbo-instruct
        #         """
        #     )
        return input_cost + output_cost


if __name__ == "__main__":
    from utils import setup_logger

    logger = setup_logger(__name__)
    user_prompt = """
    How does the prevalence of diabetes vary across different age groups and ethnicities within our patient population?
    """
    # user_prompt = "Can you list all past and current medical conditions for a given patient, including dates of diagnosis and resolution, if applicable?"
    # user_prompt = "How many male patients have diabetes alongwith hypertension?"
    vector_store = "weaviate"
    embed_model = "text-embedding-3-small"
    model = "claude-3-haiku-20240307"
    # model = "gpt-4-0125-preview"
    db_path = "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/src/patient_health_data.db"
    with open(
        "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/context_claude.txt"
    ) as f:
        context_prompt = f.read()

    with open(
        "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/schemas_1.txt",
        "r",
    ) as f:
        schemas = f.read()

    handler = LLMQueryHandler(
        model=model,
        vector_store=vector_store,
        embed_model=embed_model,
        db_path=db_path,
        top_k=3,
    )
    schemas = handler.get_semantic_schemas(user_prompt)
    df, output = handler.process_user_query(schemas, user_prompt, context_prompt)

    logger.info("Table:")
    logger.info(df.to_string())
    logger.info(f"Final SQL Query: \n\n {output['SQL_QUERY']}")
    for idx, schema in enumerate(schemas):
        logger.info(f"{idx}: {schema}")
