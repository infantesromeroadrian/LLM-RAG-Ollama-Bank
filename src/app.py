# 1. app.py

import streamlit as st
import os
from src.models.rag_system import RAGSystem
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.document_loader import DocumentLoader
from src.utils.document_processor import DataProcessor
from src.features.custom_retriever import CustomRetriever
from src.models.qa_system import QASystem

class StreamlitRAGSystem(RAGSystem):
    def __init__(self, base_dir: str, model_name: str = "llama3", temperature: float = 0.7, chunk_size: int = 2000,
                 chunk_overlap: int = 500):
        super().__init__(
            pdf_directory=os.path.join(base_dir, "GuideLines"),
            csv_file=os.path.join(base_dir, "raw_data", "BankCustomerChurnPrediction.csv"),
            base_dir=base_dir
        )
        self.base_dir = base_dir
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = Ollama(model=self.model_name, temperature=self.temperature)
        self.embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.qa_system = None

    def load_existing_vectorstore(self):
        persist_directory = os.path.join(self.base_dir, "bank_data_db")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embed_model,
            collection_name="bank_regulations_and_data"
        )
        return vectorstore

    def setup(self):
        vector_store = self.load_existing_vectorstore()
        custom_retriever = CustomRetriever(vectorstore=vector_store)
        self.qa_system = QASystem(self.llm, custom_retriever)
        self.qa_system.setup_qa_chain()
        print("QA system set up successfully")

    def update_parameters(self, model_name: str, temperature: float, chunk_size: int, chunk_overlap: int):
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = Ollama(model=self.model_name, temperature=self.temperature)
        self.reprocess_documents()

    def reprocess_documents(self):
        pdf_directory = os.path.join(self.base_dir, "GuideLines")
        csv_file = os.path.join(self.base_dir, "raw_data", "BankCustomerChurnPrediction.csv")

        pdf_docs = DocumentLoader.load_pdfs(pdf_directory)
        df = DocumentLoader.load_csv(csv_file)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_docs = text_splitter.split_documents(pdf_docs)
        csv_summary_doc = DataProcessor.create_csv_summary(df)
        csv_docs = DataProcessor.create_csv_docs(df)

        all_docs = [csv_summary_doc] + csv_docs + split_docs

        persist_directory = os.path.join(self.base_dir, "bank_data_db")
        vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embed_model,
            persist_directory=persist_directory,
            collection_name="bank_regulations_and_data"
        )

        custom_retriever = CustomRetriever(vectorstore=vector_store)
        self.qa_system = QASystem(self.llm, custom_retriever)
        self.qa_system.setup_qa_chain()

def main():
    st.title("Sistema de Consultas Bancarias RAG Configurable")

    base_dir = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/YouTube/LangChainRAGOllama/data"

    @st.cache_resource
    def load_rag_system():
        rag_system = StreamlitRAGSystem(base_dir=base_dir)
        rag_system.setup()
        return rag_system

    rag_system = load_rag_system()

    st.sidebar.header("Configuración del Sistema")
    model_name = st.sidebar.selectbox("Modelo", ["llama3", "llama2", "mistral"], index=0)
    temperature = st.sidebar.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    chunk_size = st.sidebar.number_input("Tamaño de Chunk", min_value=500, max_value=5000, value=2000, step=100)
    chunk_overlap = st.sidebar.number_input("Superposición de Chunk", min_value=0, max_value=1000, value=500, step=50)

    if st.sidebar.button("Actualizar Configuración"):
        with st.spinner("Actualizando configuración y reprocesando documentos..."):
            rag_system.update_parameters(model_name, temperature, chunk_size, chunk_overlap)
        st.success("Configuración actualizada y documentos reprocesados")

    st.write("Este sistema utiliza RAG (Retrieval-Augmented Generation) para responder preguntas sobre datos bancarios.")

    user_question = st.text_input("Ingrese su pregunta aquí:")

    if st.button("Obtener Respuesta"):
        if user_question:
            with st.spinner('Buscando la respuesta...'):
                try:
                    response = rag_system.ask_question(user_question)
                    st.write("Respuesta:")
                    st.write(response["result"])
                    st.write("\nFuente principal:")
                    source = response['source_documents'][0].metadata.get('source', 'No especificada')
                    st.write(f"Fuente: {source}")
                    st.write(response['source_documents'][0].page_content[:200])  # Primeros 200 caracteres
                except Exception as e:
                    st.error(f"Error al procesar la pregunta: {str(e)}")
        else:
            st.warning("Por favor, ingrese una pregunta.")

    st.write("\nEjemplos de preguntas que puedes hacer:")
    example_questions = [
        "¿Cuántos clientes únicos tenemos en el banco según nuestra data en el CSV?",
        "¿Cuál es el saldo promedio de los clientes?",
        "¿Cuántos países están representados en nuestros datos de clientes?",
        "¿Cuál es la tasa de abandono de clientes?",
        "¿Cuál es el rango de edades de nuestros clientes?",
        "¿Qué porcentaje de clientes tiene tarjeta de crédito?"
    ]
    for question in example_questions:
        st.write(f"- {question}")

if __name__ == "__main__":
    main()