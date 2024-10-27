# 1. app.py

# app.py

import streamlit as st
import os
import io
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.model.rag_system import RAGSystem, CustomRetriever
from src.utils.test_runner import run_tests, flatten_dict
from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.model.rag_system import RAGSystem
from src.utils.test_runner import run_tests
from src.utils.document_loader import DocumentLoader
from src.utils.document_processor import DataProcessor
from src.model.qa_system import QASystem


class StreamlitRAGSystem(RAGSystem):
    def __init__(self, base_dir: str, model_name: str = "llama3", temperature: float = 0.7,
                 chunk_size: int = 2000, chunk_overlap: int = 500):
        """
        Inicializa el sistema RAG para Streamlit.

        Args:
            base_dir (str): Directorio base para los archivos del sistema.
            model_name (str): Nombre del modelo de lenguaje a utilizar.
            temperature (float): Temperatura para la generación de texto.
            chunk_size (int): Tamaño de los chunks para dividir documentos.
            chunk_overlap (int): Superposición entre chunks.
        """
        pdf_directory = os.path.join(base_dir, "GuideLines")
        csv_file = os.path.join(base_dir, "raw_data", "BankCustomerChurnPrediction.csv")
        super().__init__(pdf_directory, csv_file, base_dir)

        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = Ollama(model=model_name, temperature=temperature)
        self.embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.qa_system = None
        self.update_llm()
        self.run()

    def update_llm(self):
        """Actualiza el modelo de lenguaje con los parámetros actuales."""
        self.llm = Ollama(model=self.model_name, temperature=self.temperature)

    def update_parameters(self, model_name: str, temperature: float, chunk_size: int, chunk_overlap: int):
        """
        Actualiza los parámetros del sistema y reprocesa los documentos.

        Args:
            model_name (str): Nuevo nombre del modelo.
            temperature (float): Nueva temperatura.
            chunk_size (int): Nuevo tamaño de chunk.
            chunk_overlap (int): Nueva superposición de chunks.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.update_llm()
        self.run()

    def run(self):
        """
        Ejecuta el flujo completo del sistema RAG.
        """
        # Cargar documentos
        pdf_docs = DocumentLoader.load_pdfs(self.pdf_directory)
        df = DocumentLoader.load_csv(self.csv_file)

        # Procesar documentos
        split_docs = DataProcessor.split_documents(pdf_docs, self.chunk_size, self.chunk_overlap)
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


@st.cache_resource
def load_rag_system(base_dir, model_name, temperature, chunk_size, chunk_overlap):
    """
    Carga y configura el sistema RAG, utilizando caché de Streamlit para mejorar el rendimiento.

    Returns:
        StreamlitRAGSystem: Instancia configurada del sistema RAG.
    """
    try:
        rag_system = StreamlitRAGSystem(base_dir=base_dir, model_name=model_name,
                                        temperature=temperature, chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap)
        return rag_system
    except Exception as e:
        st.error(f"Error al cargar el sistema RAG: {str(e)}")
        return None


def get_response(rag_system, question):
    """
    Obtiene la respuesta a una pregunta utilizando el sistema RAG.

    Args:
        rag_system (StreamlitRAGSystem): Instancia del sistema RAG.
        question (str): Pregunta a realizar.
    """
    if rag_system is None:
        st.error("El sistema RAG no está disponible debido a un error de configuración.")
        return

    if question:
        with st.spinner('Buscando la respuesta...'):
            try:
                response = rag_system.ask_question(question)
                st.write("Respuesta:")
                st.write(response["result"])
                st.write("\nFuente principal:")
                source = response['source_documents'][0].metadata.get('source', 'No especificada')
                st.write(f"Fuente: {source}")
                st.write(response['source_documents'][0].page_content[:200])  # Primeros 200 caracteres
            except Exception as e:
                st.error(f"Error al procesar la pregunta: {str(e)}")
                st.error(
                    "Si el error persiste, intente actualizar o reinstalar el modelo usando 'ollama pull [nombre_del_modelo]'")
    else:
        st.warning("Por favor, ingrese una pregunta.")


def main():
    """Función principal que configura y ejecuta la interfaz de Streamlit."""
    st.title("Sistema de Consultas Bancarias RAG Configurable")

    base_dir = "../data"

    # Configuración de la barra lateral
    st.sidebar.header("Configuración del Sistema")
    model_name = st.sidebar.selectbox("Modelo", ["llama3", "llama2", "mistral"], index=0)
    temperature = st.sidebar.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    chunk_size = st.sidebar.number_input("Tamaño de Chunk", min_value=500, max_value=5000, value=2000, step=100)
    chunk_overlap = st.sidebar.number_input("Superposición de Chunk", min_value=0, max_value=1000, value=500, step=50)

    # Verificamos si algún parámetro ha cambiado
    params_changed = (
            "model_name" not in st.session_state or
            st.session_state.model_name != model_name or
            st.session_state.temperature != temperature or
            st.session_state.chunk_size != chunk_size or
            st.session_state.chunk_overlap != chunk_overlap
    )

    # Actualizamos el estado de la sesión y recargar el sistema RAG si es necesario
    if params_changed:
        st.session_state.model_name = model_name
        st.session_state.temperature = temperature
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

        with st.spinner("Actualizando configuración y reprocesando documentos..."):
            rag_system = load_rag_system(base_dir, model_name, temperature, chunk_size, chunk_overlap)

        if rag_system:
            st.success("Configuración actualizada y documentos reprocesados")
            # Obtener una respuesta automáticamente después de la actualización
            if 'last_question' in st.session_state and st.session_state.last_question:
                get_response(rag_system, st.session_state.last_question)
    else:
        rag_system = load_rag_system(base_dir, model_name, temperature, chunk_size, chunk_overlap)

    if rag_system:
        st.write(
            "Este sistema utiliza RAG (Retrieval-Augmented Generation) para responder preguntas sobre datos bancarios y normativas.")

        # Área de preguntas y respuestas
        user_question = st.text_input("Ingrese su pregunta aquí:")

        if st.button("Obtener Respuesta") or (
                user_question and user_question != st.session_state.get('last_question', '')):
            st.session_state.last_question = user_question
            get_response(rag_system, user_question)

        # Ejemplos de preguntas
        st.write("\nEjemplos de preguntas que puedes hacer:")
        example_questions = [
            "¿Cuántos clientes únicos tenemos en el banco según nuestra data en el CSV?",
            "¿Cuál es el saldo promedio de los clientes?",
            "¿Cuántos países están representados en nuestros datos de clientes?",
            "¿Cuál es la tasa de abandono de clientes?",
            "¿Cuál es el rango de edades de nuestros clientes?",
            "¿Qué porcentaje de clientes tiene tarjeta de crédito?",
            "¿Cuáles son los requisitos para abrir una cuenta bancaria?",
            "¿Qué medidas de seguridad se aplican para proteger las transacciones en línea?",
            "¿Cuál es la política del banco respecto a los préstamos hipotecarios?",
            "¿Cómo maneja el banco las reclamaciones de los clientes?",
            "¿Cuáles son las normativas sobre prevención de lavado de dinero que sigue el banco?"
        ]
        for question in example_questions:
            if st.button(question):
                st.session_state.last_question = question
                get_response(rag_system, question)

        # Ejecución de pruebas y visualización de resultados
        if st.button("Ejecutar Pruebas"):
            with st.spinner('Ejecutando pruebas...'):
                reference_answers = [
                    "Según los datos del CSV, tenemos 10000 clientes únicos en el banco.",
                    "El saldo promedio de los clientes es 76,485.89 euros.",
                    "En nuestros datos de clientes están representados 3 países: Francia, España y Alemania.",
                    "La tasa de abandono de clientes es del 20.37%.",
                    "El rango de edades de nuestros clientes es de 18 a 92 años.",
                    "El 70.51% de los clientes tiene tarjeta de crédito.",
                    "Para abrir una cuenta bancaria, generalmente se requiere una identificación válida, comprobante de domicilio y un depósito inicial mínimo.",
                    "El banco utiliza encriptación de extremo a extremo, autenticación de dos factores y monitoreo constante para proteger las transacciones en línea.",
                    "La política de préstamos hipotecarios del banco incluye evaluación de crédito, tasas de interés competitivas y plazos flexibles de hasta 30 años.",
                    "El banco maneja las reclamaciones a través de un proceso estructurado que incluye recepción, investigación, resolución y seguimiento, con un plazo máximo de respuesta de 15 días hábiles.",
                    "El banco sigue estrictas normativas de KYC (Know Your Customer) y realiza monitoreo constante de transacciones para prevenir el lavado de dinero."
                ]
                results = run_tests(rag_system, example_questions, reference_answers, 'resultados_pruebas.csv')
            st.success("Pruebas completadas. Resultados guardados en 'resultados_pruebas.csv'")

            # Mostrar los resultados en la interfaz
            if os.path.exists('resultados_pruebas.csv'):
                df = pd.read_csv('resultados_pruebas.csv')
                st.dataframe(df)

                # Visualizaciones de las métricas
                st.subheader("Visualización de Métricas")

                # Crear subplots con Plotly
                fig = make_subplots(rows=3, cols=1, subplot_titles=(
                'BLEU y ROUGE Scores', 'Tiempo de Respuesta', 'Relevancia de la Fuente vs BLEU Score'))

                # Gráfico de barras para BLEU y ROUGE scores
                for column in ['bleu_score', 'rouge_scores_rouge-1', 'rouge_scores_rouge-2', 'rouge_scores_rouge-l']:
                    fig.add_trace(go.Bar(x=df.index, y=df[column], name=column), row=1, col=1)

                # Gráfico de líneas para response_time
                fig.add_trace(go.Scatter(x=df.index, y=df['response_time'], mode='lines+markers', name='Response Time'),
                              row=2, col=1)

                # Gráfico de dispersión para source_relevance vs bleu_score
                fig.add_trace(go.Scatter(x=df['source_relevance'], y=df['bleu_score'], mode='markers',
                                         name='Source Relevance vs BLEU Score'), row=3, col=1)

                # Actualizar el diseño
                fig.update_layout(height=900, width=800, title_text="Métricas de Evaluación")
                fig.update_xaxes(title_text="Preguntas", row=1, col=1)
                fig.update_xaxes(title_text="Preguntas", row=2, col=1)
                fig.update_xaxes(title_text="Relevancia de la Fuente", row=3, col=1)
                fig.update_yaxes(title_text="Scores", row=1, col=1)
                fig.update_yaxes(title_text="Tiempo (s)", row=2, col=1)
                fig.update_yaxes(title_text="BLEU Score", row=3, col=1)

                # Mostrar el gráfico
                st.plotly_chart(fig)

    else:
        st.error("El sistema RAG no está disponible debido a un error de configuración.")
        st.warning("Por favor, verifique la configuración y vuelva a intentarlo.")
        st.stop()

if __name__ == "__main__":
    main()