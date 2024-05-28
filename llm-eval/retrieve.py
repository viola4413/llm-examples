from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

pc = Pinecone(api_key=pinecone_api_key)
embed_model = HuggingFaceEmbedding("Snowflake/snowflake-arctic-embed-m")

def retrieve(query: str):
    index_name = "streamlit-docs"
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    retriever = index.as_retriever()
    return retriever.retrieve(query)