from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import os

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_SERVERLESS_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

with open(
    "/Users/karankinariwala/Library/CloudStorage/OneDrive-Personal/Medeva LLM Internship/data/context.txt",
    "r",
) as f:
    text = f.read()

schemas = text.split("&")
for i in range(len(schemas)):
    print(schemas[i].split(" ")[2])

parser = SimpleNodeParser()
docs = []
for i, schema in enumerate(schemas):
    docs.append(
        Document(text=schema, doc_id=i, extra_info={"title": schema.split(" ")[2]})
    )

nodes = parser.get_nodes_from_documents(docs)

pc = Pinecone(api_key=pinecone_api_key)

try:
    pc_index = pc.create_index(
        name="schema-index",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )
except:
    print("Vector Index Already Exists")

vector_store = PineconeVectorStore(
    pinecone_index=pc_index, index_name="schema-index", api_key=pinecone_api_key
)
