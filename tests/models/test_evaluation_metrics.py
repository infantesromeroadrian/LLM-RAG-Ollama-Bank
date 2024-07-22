# test_evaluation_metrics.py

import unittest
from src.models.evaluation_metrics import EvaluationMetrics


class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        self.metrics = EvaluationMetrics()

    def test_calculate_bleu(self):
        reference = "The cat is on the mat"
        candidate = "The cat is on the mat"
        self.assertAlmostEqual(self.metrics.calculate_bleu(reference, candidate), 1.0)

        candidate = "The dog is in the house"
        self.assertLess(self.metrics.calculate_bleu(reference, candidate), 0.5)

        # Prueba con cadenas vacías
        self.assertEqual(self.metrics.calculate_bleu("", ""), 0)
        self.assertEqual(self.metrics.calculate_bleu(reference, ""), 0)

    def test_calculate_rouge(self):
        reference = "The cat is on the mat"
        candidate = "The cat is on the mat"
        scores = self.metrics.calculate_rouge(reference, candidate)
        self.assertAlmostEqual(scores['rouge-1'], 1.0)
        self.assertAlmostEqual(scores['rouge-2'], 1.0)
        self.assertAlmostEqual(scores['rouge-l'], 1.0)

        candidate = "The dog is in the house"
        scores = self.metrics.calculate_rouge(reference, candidate)
        self.assertLess(scores['rouge-1'], 0.5)
        self.assertLess(scores['rouge-2'], 0.5)
        self.assertLess(scores['rouge-l'], 0.5)

        # Prueba con cadenas vacías
        scores = self.metrics.calculate_rouge("", "")
        self.assertEqual(scores['rouge-1'], 0)
        self.assertEqual(scores['rouge-2'], 0)
        self.assertEqual(scores['rouge-l'], 0)

    def test_evaluate_source_relevance(self):
        question = "What is the capital of France?"
        source_content = "Paris is the capital and most populous city of France."
        self.assertGreater(self.metrics.evaluate_source_relevance(question, source_content), 0.5)

        source_content = "The Eiffel Tower is located in Paris."
        self.assertLess(self.metrics.evaluate_source_relevance(question, source_content), 0.5)

        # Prueba con cadenas vacías
        self.assertEqual(self.metrics.evaluate_source_relevance("", ""), 0)
        self.assertEqual(self.metrics.evaluate_source_relevance(question, ""), 0)

    def test_evaluate_response(self):
        question = "What is the capital of France?"
        answer = "The capital of France is Paris."
        reference_answer = "Paris is the capital of France."
        source_content = "Paris is the capital and most populous city of France."

        evaluation = self.metrics.evaluate_response(question, answer, reference_answer, source_content)

        self.assertIn('bleu_score', evaluation)
        self.assertIn('rouge_scores', evaluation)
        self.assertIn('source_relevance', evaluation)
        self.assertGreater(evaluation['bleu_score'], 0.5)
        self.assertGreater(evaluation['rouge_scores']['rouge-1'], 0.5)
        self.assertGreater(evaluation['source_relevance'], 0.5)

    def test_edge_cases(self):
        # Probar con entradas muy largas
        long_text = " ".join(["word"] * 1000)
        self.assertIsNotNone(self.metrics.calculate_bleu(long_text, long_text))
        self.assertIsNotNone(self.metrics.calculate_rouge(long_text, long_text))

        # Probar con caracteres especiales
        special_text = "!@#$%^&*()_+{}|:<>?~`-=[]\\;',./'"
        self.assertIsNotNone(self.metrics.calculate_bleu(special_text, special_text))
        self.assertIsNotNone(self.metrics.calculate_rouge(special_text, special_text))


if __name__ == '__main__':
    unittest.main()