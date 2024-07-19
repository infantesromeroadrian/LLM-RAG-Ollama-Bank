# 7. rag_system.py

import os
from src.utils.document_manager import DocumentManager
from src.features.vector_store_manager import VectorStoreManager
from src.features.custom_retriever import CustomRetriever
from src.models.qa_system import QASystem
from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

class RAGSystem:
    def __init__(self, pdf_directory: str, csv_file: str, base_dir: str):
        self.document_manager = DocumentManager(pdf_directory, csv_file)
        self.llm = Ollama(model="llama3")
        self.embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store_manager = VectorStoreManager(self.embed_model, base_dir)
        self.qa_system = None

    def run(self):
        # Load and process documents
        documents = self.document_manager.load_and_process_documents()
        print(f"Total documents processed: {len(documents)}")

        # Create vector store
        vector_store = self.vector_store_manager.create_vector_store(documents)
        print(f"Vector store created with {vector_store._collection.count()} documents")

        # Set up custom retriever
        custom_retriever = CustomRetriever(vectorstore=vector_store)

        # Set up QA system
        self.qa_system = QASystem(self.llm, custom_retriever)
        self.qa_system.setup_qa_chain()
        print("QA system set up successfully")

    def ask_question(self, question: str):
        if not self.qa_system:
            raise ValueError("QA system not set up. Run the 'run' method first.")
        return self.qa_system.ask_question(question)