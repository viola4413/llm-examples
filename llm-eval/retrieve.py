
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st
import os

class PineconeRetriever:
    def __init__(self):
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.embed_model = HuggingFaceEmbedding("Snowflake/snowflake-arctic-embed-m")
        self.index_name = "streamlit-docs"

    def retrieve(self, query: str):
        pinecone_index = self.pc.Index(self.index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=self.embed_model)
        retriever = index.as_retriever()
        return retriever.retrieve(query)
    
    def set_api_key(self, api_key: str):
        self.pinecone_api_key = api_key
        self.pc = Pinecone(api_key=self.pinecone_api_key)
