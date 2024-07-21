# 9. test_runner.py

import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
from src.models.evaluation_metrics import EvaluationMetrics

def flatten_dict(d, parent_key='', sep='_'):
    """
    Aplana un diccionario anidado.

    Args:
        d (dict): El diccionario a aplanar.
        parent_key (str): La clave padre para la recursión.
        sep (str): El separador a usar entre las claves.

    Returns:
        dict: El diccionario aplanado.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def run_tests(rag_system, questions, reference_answers, output_file='test_results.csv'):
    """
    Ejecuta pruebas en el sistema RAG y guarda los resultados.

    Args:
        rag_system (RAGSystem): El sistema RAG a probar.
        questions (list): Lista de preguntas de prueba.
        reference_answers (list): Lista de respuestas de referencia.
        output_file (str): Nombre del archivo CSV de salida.

    Returns:
        list: Lista de resultados de las pruebas.
    """
    metrics = EvaluationMetrics()
    results = []

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'question', 'answer', 'reference_answer', 'source', 'source_content',
                      'bleu_score', 'rouge_scores_rouge-1', 'rouge_scores_rouge-2', 'rouge_scores_rouge-l', 'source_relevance', 'response_time']
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
                source_content = response['source_documents'][0].page_content[:200]

                print(f"Respuesta del modelo: {answer}")
                print(f"Fuente: {source}")
                print(f"Contenido de la fuente: {source_content}")

                evaluation_results = metrics.evaluate_response(
                    question,
                    answer,
                    reference_answer,
                    source_content
                )

                evaluation_results.update({
                    'timestamp': datetime.now().isoformat(),
                    'question': question,
                    'answer': answer,
                    'reference_answer': reference_answer,
                    'source': source,
                    'source_content': source_content,
                    'response_time': response_time
                })

                flattened_results = flatten_dict(evaluation_results)
                results.append(flattened_results)
                writer.writerow(flattened_results)

            except Exception as e:
                print(f"Error al procesar la pregunta: {str(e)}")

    print(f"Resultados guardados en {output_file}")
    return results

def plot_results(results):
    """
    Genera gráficos de los resultados de las pruebas.

    Args:
        results (list): Lista de resultados de las pruebas.
    """
    bleu_scores = [res['bleu_score'] for res in results]
    rouge_1_scores = [res['rouge_scores_rouge-1'] for res in results]
    rouge_2_scores = [res['rouge_scores_rouge-2'] for res in results]
    rouge_l_scores = [res['rouge_scores_rouge-l'] for res in results]
    source_relevance = [res['source_relevance'] for res in results]
    response_times = [res['response_time'] for res in results]
    questions = [res['question'] for res in results]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(questions, bleu_scores, label='BLEU Score')
    plt.plot(questions, rouge_1_scores, label='ROUGE-1 Score')
    plt.plot(questions, rouge_2_scores, label='ROUGE-2 Score')
    plt.plot(questions, rouge_l_scores, label='ROUGE-L Score')
    plt.plot(questions, source_relevance, label='Source Relevance')
    plt.ylabel('Scores')
    plt.legend()
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    plt.plot(questions, response_times, label='Response Time', color='r')
    plt.ylabel('Response Time (s)')
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()