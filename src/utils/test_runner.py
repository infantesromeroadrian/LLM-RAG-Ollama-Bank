# 9. test_runner.py

"""
test_runner.py

Este módulo contiene la función run_tests para ejecutar pruebas automatizadas
en el sistema RAG (Retrieval-Augmented Generation) y evaluar su rendimiento.

El módulo proporciona una función para ejecutar una serie de preguntas de prueba,
evaluar las respuestas utilizando múltiples métricas, y guardar los resultados
en un archivo CSV para su posterior análisis.

Dependencias:
- csv: Para escribir los resultados en formato CSV.
- datetime: Para registrar el timestamp de cada prueba.
- time: Para medir el tiempo de respuesta.
- src.models.evaluation_metrics: Para utilizar la clase EvaluationMetrics.

Funciones:
- run_tests: Ejecuta las pruebas en el sistema RAG y guarda los resultados.
"""

import csv
from datetime import datetime
import time
from src.models.evaluation_metrics import EvaluationMetrics

def run_tests(rag_system, questions, reference_answers, output_file='test_results.csv'):
    """
    Ejecuta pruebas en el sistema RAG y guarda los resultados en un archivo CSV.

    Esta función realiza las siguientes operaciones para cada pregunta:
    1. Obtiene una respuesta del sistema RAG.
    2. Mide el tiempo de respuesta.
    3. Evalúa la respuesta utilizando múltiples métricas.
    4. Guarda los resultados en un archivo CSV.

    Args:
        rag_system: Una instancia del sistema RAG a probar.
        questions (list): Lista de preguntas de prueba.
        reference_answers (list): Lista de respuestas de referencia correspondientes.
        output_file (str): Nombre del archivo CSV para guardar los resultados.

    Returns:
        None

    Raises:
        Exception: Si ocurre un error al procesar una pregunta, se captura y se imprime.

    Note:
        El archivo CSV de salida incluirá las siguientes columnas:
        timestamp, question, answer, reference_answer, source, source_content,
        bleu_score, rouge-1, rouge-2, rouge-l, perplexity, source_relevance, response_time
    """
    metrics = EvaluationMetrics()

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'question', 'answer', 'reference_answer', 'source', 'source_content',
                      'bleu_score', 'rouge-1', 'rouge-2', 'rouge-l', 'perplexity', 'source_relevance', 'response_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for question, reference_answer in zip(questions, reference_answers):
            print(f"\nPregunta: {question}")
            try:
                # Medir el tiempo de respuesta
                start_time = time.time()
                response = rag_system.ask_question(question)
                response_time = time.time() - start_time

                answer = response["result"]
                source = response['source_documents'][0].metadata.get('source', 'No especificada')
                source_content = response['source_documents'][0].page_content[:200]  # Primeros 200 caracteres

                # Evaluar la respuesta
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

                # Escribir resultados en el archivo CSV
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

# Ejemplo de uso:
# rag_system = RAGSystem(...)
# questions = ["¿Pregunta 1?", "¿Pregunta 2?", ...]
# reference_answers = ["Respuesta 1", "Respuesta 2", ...]
# run_tests(rag_system, questions, reference_answers, "resultados_test.csv")