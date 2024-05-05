from openai import OpenAI
import anthropic
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
        index_name: str = None,
        top_k=5,
    ):
        self.model = model
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.index_name = index_name
        self.top_k = top_k

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

    def generate_sql_query(
        self,
        schemas: list[str],
        user_prompt: str,
        context: str = None,
        system_prompt: str = None,
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
        if system_prompt == None:
            system_prompt = self._create_system_prompt(schemas, context)

        if self._find_model() == "gpt":
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            if self.openai_api_key is None:
                raise ValueError(
                    "OPENAI_API_KEY must be specified as an environment variable."
                )

            self.client = OpenAI(api_key=self.openai_api_key)
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

            client = anthropic.Anthropic(api_key=self.claude_api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
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

    @staticmethod
    def _create_system_prompt(schemas: list[str], context: str) -> str:
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

        {context}
        """

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
    # user_prompt = """
    # How does the prevalence of specific conditions vary across different age groups and ethnicities within our patient population?
    # """
    # user_prompt = "Can you list all past and current medical conditions for a given patient, including dates of diagnosis and resolution, if applicable?"
    user_prompt = "How many male patients have diabetes alongwith hypertension?"
    vector_store = "weaviate"
    embed_model = "text-embedding-3-small"
    model = "claude-3-haiku-20240307"
    # gpt_model = "gpt-4-0125-preview"

    with open(
        "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/context.txt"
    ) as f:
        context_prompt = f.read()

    handler = LLMQueryHandler(
        model=model, vector_store=vector_store, embed_model=embed_model, top_k=3
    )
    schemas = handler.get_semantic_schemas(user_prompt)
    output = handler.generate_sql_query(schemas, user_prompt, context=context_prompt)

    cost = handler.calculate_query_execution_cost(
        output["N_PROMPT_TOKENS"], output["N_GENERATED_TOKENS"]
    )

    logger.info(f"SQL Query: \n\n {output['SQL_QUERY']}")
    for idx, schema in enumerate(schemas):
        logger.info(idx)
        logger.info(schema)
    logger.info(f"Cost = ${cost:.5f}")
    logger.info(f"Model: {output['MODEL']}")
    logger.info(f"Number of Prompt Tokens: {output['N_PROMPT_TOKENS']}")
    logger.info(f"Number of Generated Tokens: {output['N_GENERATED_TOKENS']}")
