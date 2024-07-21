# 7. rag_system.py

from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.utils.document_loader import DocumentLoader
from src.utils.document_processor import DataProcessor
from src.features.vector_store_manager import VectorStoreManager
from src.features.custom_retriever import CustomRetriever
from src.models.qa_system import QASystem

class RAGSystem:
    """
    Sistema RAG que integra la gestión de documentos, almacenamiento de vectores,
    recuperación personalizada y un sistema de preguntas y respuestas.

    Atributos:
        pdf_directory (str): Ruta al directorio que contiene los archivos PDF.
        csv_file (str): Ruta al archivo CSV con datos.
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
        self.pdf_directory = pdf_directory
        self.csv_file = csv_file
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
        # Cargar documentos
        pdf_docs = DocumentLoader.load_pdfs(self.pdf_directory)
        df = DocumentLoader.load_csv(self.csv_file)

        # Procesar documentos
        split_docs = DataProcessor.split_documents(pdf_docs)
        csv_summary_doc = DataProcessor.create_csv_summary(df)
        csv_docs = DataProcessor.create_csv_docs(df)

        # Combinar todos los documentos procesados
        all_docs = [csv_summary_doc] + csv_docs + split_docs

        # Crear vector store
        vector_store = self.vector_store_manager.create_vector_store(all_docs)
        print(f"Vector store created with {vector_store._collection.count()} documents")

        # Configurar el recuperador personalizado
        custom_retriever = CustomRetriever(vectorstore=vector_store)

        # Configurar el sistema QA
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

    def get_system_info(self):
        """
        Proporciona información sobre el estado actual del sistema RAG.

        Returns:
            dict: Un diccionario con información sobre los componentes del sistema.
        """
        return {
            "pdf_directory": self.pdf_directory,
            "csv_file": self.csv_file,
            "llm_model": self.llm.model,
            "embedding_model": self.embed_model.model_name,
            "vector_store_base_dir": self.vector_store_manager.base_dir,
            "qa_system_status": "Configured" if self.qa_system else "Not configured"
        }