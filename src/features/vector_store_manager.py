# 4. vector_store_manager.py

import os
from langchain_community.vectorstores import Chroma

class VectorStoreManager:
    def __init__(self, embed_model, base_dir):
        self.embed_model = embed_model
        self.base_dir = base_dir

    def create_vector_store(self, documents):
        persist_directory = os.path.join(self.base_dir, "bank_data_db")
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embed_model,
            persist_directory=persist_directory,
            collection_name="bank_regulations_and_data"
        )