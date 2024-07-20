# 8. evaluation_metrics.py

"""
evaluation_metrics.py

Este módulo contiene la clase EvaluationMetrics, que proporciona métodos para evaluar
la calidad de las respuestas generadas por el sistema RAG (Retrieval-Augmented Generation).

Dependencias:
- numpy: Para cálculos numéricos eficientes.
- collections: Para el uso de Counter en el cálculo de BLEU.
- rouge: Para el cálculo de métricas ROUGE.

La clase EvaluationMetrics incluye métodos para calcular BLEU, ROUGE, perplexidad,
y relevancia de la fuente, así como un método general para evaluar respuestas.
"""

import numpy as np
from collections import Counter
from rouge import Rouge

class EvaluationMetrics:
    """
    Clase que proporciona métodos para evaluar la calidad de las respuestas generadas.

    Esta clase implementa varias métricas de evaluación comúnmente utilizadas en el
    procesamiento de lenguaje natural y sistemas de pregunta-respuesta.
    """

    def __init__(self):
        """
        Inicializa la clase EvaluationMetrics.

        Configura el objeto Rouge para el cálculo de métricas ROUGE.
        """
        self.rouge = Rouge()

    def calculate_bleu(self, reference, candidate, max_n=4):
        """
        Calcula una versión simplificada del score BLEU.

        Args:
            reference (str): La respuesta de referencia.
            candidate (str): La respuesta generada por el sistema.
            max_n (int): El n-grama máximo a considerar (por defecto 4).

        Returns:
            float: El score BLEU calculado.
        """
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
        """
        Calcula los scores ROUGE.

        Args:
            reference (str): La respuesta de referencia.
            candidate (str): La respuesta generada por el sistema.

        Returns:
            dict: Un diccionario con los scores ROUGE-1, ROUGE-2 y ROUGE-L.
        """
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
        """
        Calcula la perplejidad (implementación simplificada).

        Args:
            llm: El modelo de lenguaje utilizado para calcular la perplejidad.
            text (str): El texto para el cual se calcula la perplejidad.

        Returns:
            float: La perplejidad calculada o infinito en caso de error.
        """
        try:
            logprobs = llm.get_logprobs(text)
            return np.exp(-np.mean(logprobs))
        except Exception as e:
            print(f"Error al calcular la perplejidad: {e}")
            return float('inf')  # Devuelve infinito en caso de error

    def evaluate_source_relevance(self, question, source_content):
        """
        Evalúa la relevancia de la fuente respecto a la pregunta.

        Args:
            question (str): La pregunta realizada.
            source_content (str): El contenido de la fuente utilizada para la respuesta.

        Returns:
            float: Un score de relevancia entre 0 y 1.
        """
        question_words = set(question.lower().split())
        source_words = set(source_content.lower().split())
        common_words = question_words.intersection(source_words)
        return len(common_words) / len(question_words) if question_words else 0

    def evaluate_response(self, question, answer, reference_answer, source_content, llm):
        """
        Evalúa una respuesta utilizando todas las métricas implementadas.

        Args:
            question (str): La pregunta realizada.
            answer (str): La respuesta generada por el sistema.
            reference_answer (str): La respuesta de referencia.
            source_content (str): El contenido de la fuente utilizada para la respuesta.
            llm: El modelo de lenguaje utilizado para calcular la perplejidad.

        Returns:
            dict: Un diccionario con todos los scores calculados.
        """
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

# Ejemplo de uso:
# metrics = EvaluationMetrics()
# results = metrics.evaluate_response(question, answer, reference, source, llm)
# print(results)