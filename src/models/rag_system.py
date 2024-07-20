# 7. rag_system.py

"""
rag_system.py

Este módulo implementa un sistema RAG (Retrieval-Augmented Generation) que integra
la gestión de documentos, el almacenamiento de vectores, la recuperación personalizada
y un sistema de preguntas y respuestas.

Dependencias:
- os
- src.utils.document_manager
- src.features.vector_store_manager
- src.features.custom_retriever
- src.models.qa_system
- langchain_community.llms
- langchain_community.embeddings.fastembed

Clases:
- RAGSystem
"""

import os
from src.utils.document_manager import DocumentManager
from src.features.vector_store_manager import VectorStoreManager
from src.features.custom_retriever import CustomRetriever
from src.models.qa_system import QASystem
from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

class RAGSystem:
    """
    Sistema RAG que integra la gestión de documentos, almacenamiento de vectores,
    recuperación personalizada y un sistema de preguntas y respuestas.

    Esta clase orquesta el flujo completo de un sistema RAG, desde la carga de documentos
    hasta la generación de respuestas a preguntas.

    Atributos:
        document_manager (DocumentManager): Gestor para cargar y procesar documentos.
        llm (Ollama): Modelo de lenguaje para generar respuestas.
        embed_model (FastEmbedEmbeddings): Modelo de embeddings para vectorizar documentos.
        vector_store_manager (VectorStoreManager): Gestor del almacén de vectores.
        qa_system (QASystem): Sistema de preguntas y respuestas.
    """

    def __init__(self, pdf_directory: str, csv_file: str, base_dir: str):
        """
        Inicializa el sistema RAG.

        Args:
            pdf_directory (str): Ruta al directorio que contiene los archivos PDF.
            csv_file (str): Ruta al archivo CSV con datos.
            base_dir (str): Directorio base para almacenar el almacén de vectores.
        """
        self.document_manager = DocumentManager(pdf_directory, csv_file)
        self.llm = Ollama(model="llama3")
        self.embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store_manager = VectorStoreManager(self.embed_model, base_dir)
        self.qa_system = None

    def run(self):
        """
        Ejecuta el flujo completo del sistema RAG.

        Este método carga y procesa los documentos, crea el almacén de vectores,
        configura el recuperador personalizado y establece el sistema de preguntas y respuestas.
        """
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
        """
        Realiza una pregunta al sistema RAG.

        Args:
            question (str): La pregunta a responder.

        Returns:
            dict: Un diccionario que contiene la respuesta y los documentos fuente.

        Raises:
            ValueError: Si el sistema QA no ha sido configurado previamente.
        """
        if not self.qa_system:
            raise ValueError("QA system not set up. Run the 'run' method first.")
        return self.qa_system.ask_question(question)

# Ejemplo de uso:
# rag_system = RAGSystem(pdf_directory="/path/to/pdfs", csv_file="/path/to/data.csv", base_dir="/path/to/base")
# rag_system.run()
# response = rag_system.ask_question("¿Cuál es el saldo promedio de los clientes?")
# print(response["result"])