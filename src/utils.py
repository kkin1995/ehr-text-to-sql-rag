def check_valid_vector_store(vector_store_name):
    if vector_store_name not in ["pinecone", "weaviate"]:
        return False
    else:
        return True
