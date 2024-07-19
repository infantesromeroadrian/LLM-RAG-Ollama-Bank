# 1. app.py


import streamlit as st
import os
import pandas as pd
import csv
from datetime import datetime
import numpy as np
from collections import Counter
from rouge import Rouge
import time
from src.models.rag_system import RAGSystem
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.document_loader import DocumentLoader
from src.utils.document_processor import DataProcessor
from src.features.custom_retriever import CustomRetriever
from src.models.qa_system import QASystem


class EvaluationMetrics:
    def __init__(self):
        self.rouge = Rouge()

    def calculate_bleu(self, reference, candidate, max_n=4):
        """Calcula una versión simplificada del score BLEU."""

        def count_ngrams(sentence, n):
            return Counter(zip(*[sentence[i:] for i in range(n)]))

        reference_words = reference.split()
        candidate_words = candidate.split()

        scores = []
        for n in range(1, min(max_n, len(candidate_words)) + 1):
            ref_ngrams = count_ngrams(reference_words, n)
            cand_ngrams = count_ngrams(candidate_words, n)

            matches = sum((ref_ngrams & cand_ngrams).values())
            total = sum(cand_ngrams.values())

            score = matches / total if total > 0 else 0
            scores.append(score)

        return np.mean(scores) if scores else 0

    def calculate_rouge(self, reference, candidate):
        """Calcula los scores ROUGE."""
        try:
            scores = self.rouge.get_scores(candidate, reference)
            return {
                'rouge-1': scores[0]['rouge-1']['f'],
                'rouge-2': scores[0]['rouge-2']['f'],
                'rouge-l': scores[0]['rouge-l']['f']
            }
        except Exception as e:
            print(f"Error al calcular ROUGE: {e}")
            return {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

    def calculate_perplexity(self, llm, text):
        """Calcula la perplejidad (implementación simplificada)."""
        try:
            logprobs = llm.get_logprobs(text)
            return np.exp(-np.mean(logprobs))
        except Exception as e:
            print(f"Error al calcular la perplejidad: {e}")
            return float('inf')  # Devuelve infinito en caso de error

    def evaluate_source_relevance(self, question, source_content):
        """Evalúa la relevancia de la fuente respecto a la pregunta."""
        question_words = set(question.lower().split())
        source_words = set(source_content.lower().split())
        common_words = question_words.intersection(source_words)
        return len(common_words) / len(question_words) if question_words else 0

    def evaluate_response(self, question, answer, reference_answer, source_content, llm):
        """Evalúa una respuesta utilizando todas las métricas."""
        bleu_score = self.calculate_bleu(reference_answer, answer)
        rouge_scores = self.calculate_rouge(reference_answer, answer)
        perplexity = self.calculate_perplexity(llm, answer)
        source_relevance = self.evaluate_source_relevance(question, source_content)

        return {
            'bleu_score': bleu_score,
            'rouge_scores': rouge_scores,
            'perplexity': perplexity,
            'source_relevance': source_relevance
        }


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


def run_tests(rag_system, questions, reference_answers, output_file='test_results.csv'):
    metrics = EvaluationMetrics()

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'question', 'answer', 'reference_answer', 'source', 'source_content',
                      'bleu_score', 'rouge-1', 'rouge-2', 'rouge-l', 'perplexity', 'source_relevance', 'response_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for question, reference_answer in zip(questions, reference_answers):
            print(f"\nPregunta: {question}")
            try:
                start_time = time.time()
                response = rag_system.ask_question(question)
                response_time = time.time() - start_time

                answer = response["result"]
                source = response['source_documents'][0].metadata.get('source', 'No especificada')
                source_content = response['source_documents'][0].page_content[:200]  # Primeros 200 caracteres

                evaluation_results = metrics.evaluate_response(
                    question,
                    answer,
                    reference_answer,
                    source_content,
                    rag_system.llm
                )

                print("Respuesta del modelo:")
                print(answer)
                print("\nMétricas de evaluación:")
                print(evaluation_results)

                # Escribir en el archivo CSV
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'question': question,
                    'answer': answer,
                    'reference_answer': reference_answer,
                    'source': source,
                    'source_content': source_content,
                    'bleu_score': evaluation_results['bleu_score'],
                    'rouge-1': evaluation_results['rouge_scores']['rouge-1'],
                    'rouge-2': evaluation_results['rouge_scores']['rouge-2'],
                    'rouge-l': evaluation_results['rouge_scores']['rouge-l'],
                    'perplexity': evaluation_results['perplexity'],
                    'source_relevance': evaluation_results['source_relevance'],
                    'response_time': response_time
                })

            except Exception as e:
                print(f"Error al procesar la pregunta: {str(e)}")

    print(f"Resultados guardados en {output_file}")


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

    st.write(
        "Este sistema utiliza RAG (Retrieval-Augmented Generation) para responder preguntas sobre datos bancarios.")

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
            run_tests(rag_system, example_questions, reference_answers, 'resultados_pruebas.csv')
        st.success("Pruebas completadas. Resultados guardados en 'resultados_pruebas.csv'")

        # Mostrar los resultados en la interfaz
        if os.path.exists('resultados_pruebas.csv'):
            df = pd.read_csv('resultados_pruebas.csv')
            st.dataframe(df)

            # Añadir visualizaciones de las métricas
            st.subheader("Visualización de Métricas")

            # Gráfico de barras para BLEU y ROUGE scores
            st.bar_chart(df[['bleu_score', 'rouge-1', 'rouge-2', 'rouge-l']])

            # Gráfico de líneas para perplexity y response_time
            st.line_chart(df[['perplexity', 'response_time']])

            # Gráfico de dispersión para source_relevance vs bleu_score
            st.scatter_chart(df[['source_relevance', 'bleu_score']])


if __name__ == "__main__":
    main()