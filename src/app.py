# 1. app.py

import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.models.rag_system import RAGSystem, CustomRetriever
from src.utils.test_runner import run_tests, flatten_dict
from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.models.rag_system import RAGSystem
from src.utils.test_runner import run_tests


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
        self.update_llm()

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
        self.run()  # Reprocesa los documentos con los nuevos parámetros

def main():
    """Función principal que configura y ejecuta la interfaz de Streamlit."""
    st.title("Sistema de Consultas Bancarias RAG Configurable")

    base_dir = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/YouTube/LangChainRAGOllama/data"

    @st.cache_resource
    def load_rag_system():
        """
        Carga y configura el sistema RAG, utilizando caché de Streamlit para mejorar el rendimiento.

        Returns:
            StreamlitRAGSystem: Instancia configurada del sistema RAG.
        """
        rag_system = StreamlitRAGSystem(base_dir=base_dir)
        rag_system.run()
        return rag_system

    rag_system = load_rag_system()

    # Configuración de la barra lateral
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

    # Área de preguntas y respuestas
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

    # Ejemplos de preguntas
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

    # Ejecución de pruebas y visualización de resultados
    if st.button("Ejecutar Pruebas"):
        with st.spinner('Ejecutando pruebas...'):
            reference_answers = [
                "Según los datos del CSV, tenemos 10000 clientes únicos en el banco.",
                "El saldo promedio de los clientes es 76,485.89 euros.",
                "En nuestros datos de clientes están representados 3 países: Francia, España y Alemania.",
                "La tasa de abandono de clientes es del 20.37%.",
                "El rango de edades de nuestros clientes es de 18 a 92 años.",
                "El 70.51% de los clientes tiene tarjeta de crédito."
            ]
            results = run_tests(rag_system, example_questions, reference_answers, 'resultados_pruebas.csv')
        st.success("Pruebas completadas. Resultados guardados en 'resultados_pruebas.csv'")

        # Mostrar los resultados en la interfaz
        if os.path.exists('resultados_pruebas.csv'):
            df = pd.read_csv('resultados_pruebas.csv')
            st.dataframe(df)

            # Visualizaciones de las métricas
            st.subheader("Visualización de Métricas")

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

            # Gráfico de barras para BLEU y ROUGE scores
            df[['bleu_score', 'rouge_scores_rouge-1', 'rouge_scores_rouge-2', 'rouge_scores_rouge-l']].plot(kind='bar', ax=ax1)
            ax1.set_title('BLEU y ROUGE Scores')
            ax1.set_xlabel('Preguntas')
            ax1.set_ylabel('Scores')
            ax1.legend(loc='best')

            # Gráfico de líneas para perplexity y response_time
            ax2.plot(df.index, df['response_time'], label='Response Time')
            ax2.set_title('Tiempo de Respuesta')
            ax2.set_xlabel('Preguntas')
            ax2.set_ylabel('Tiempo (s)')
            ax2.legend(loc='best')

            # Gráfico de dispersión para source_relevance vs bleu_score
            ax3.scatter(df['source_relevance'], df['bleu_score'])
            ax3.set_title('Relevancia de la Fuente vs BLEU Score')
            ax3.set_xlabel('Relevancia de la Fuente')
            ax3.set_ylabel('BLEU Score')

            st.pyplot(fig)

if __name__ == "__main__":
    main()