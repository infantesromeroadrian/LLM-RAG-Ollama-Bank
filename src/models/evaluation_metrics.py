# 8. evaluation_metrics.py

from collections import Counter
import numpy as np
from rouge import Rouge

class EvaluationMetrics:
    """
    Clase para calcular métricas de evaluación para respuestas generadas.
    """

    def __init__(self):
        self.rouge = Rouge()

    def calculate_bleu(self, reference, candidate, max_n=4):
        """
        Calcula el score BLEU entre una referencia y un candidato.

        Args:
            reference (str): La respuesta de referencia.
            candidate (str): La respuesta candidata a evaluar.
            max_n (int): El n-grama máximo a considerar.

        Returns:
            float: El score BLEU.
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
        Calcula los scores ROUGE entre una referencia y un candidato.

        Args:
            reference (str): La respuesta de referencia.
            candidate (str): La respuesta candidata a evaluar.

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

    def evaluate_source_relevance(self, question, source_content):
        """
        Evalúa la relevancia de la fuente con respecto a la pregunta.

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

    def evaluate_response(self, question, answer, reference_answer, source_content):
        """
        Evalúa una respuesta generada utilizando múltiples métricas.

        Args:
            question (str): La pregunta realizada.
            answer (str): La respuesta generada.
            reference_answer (str): La respuesta de referencia.
            source_content (str): El contenido de la fuente utilizada para la respuesta.

        Returns:
            dict: Un diccionario con los scores de las diferentes métricas.
        """
        bleu_score = self.calculate_bleu(reference_answer, answer)
        rouge_scores = self.calculate_rouge(reference_answer, answer)
        source_relevance = self.evaluate_source_relevance(question, source_content)

        return {
            'bleu_score': bleu_score,
            'rouge_scores': rouge_scores,
            'source_relevance': source_relevance
        }