# test_test_runner.py

import unittest
from unittest.mock import Mock, patch, mock_open
import csv
import io
from src.utils.test_runner import flatten_dict, run_tests, plot_results

class TestTestRunner(unittest.TestCase):

    def test_flatten_dict(self):
        nested_dict = {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3
                }
            }
        }
        flattened = flatten_dict(nested_dict)
        self.assertEqual(flattened, {'a': 1, 'b_c': 2, 'b_d_e': 3})

    @patch('src.utils.test_runner.EvaluationMetrics')
    @patch('src.utils.test_runner.open', new_callable=mock_open)
    @patch('src.utils.test_runner.csv.DictWriter')
    def test_run_tests(self, mock_csv_writer, mock_file, mock_metrics):
        # Configurar mocks
        mock_rag_system = Mock()
        mock_rag_system.ask_question.return_value = {
            "result": "Test answer",
            "source_documents": [Mock(metadata={'source': 'Test source'}, page_content='Test content')]
        }

        mock_metrics_instance = mock_metrics.return_value
        mock_metrics_instance.evaluate_response.return_value = {
            'bleu_score': 0.8,
            'rouge_scores': {'rouge-1': 0.7, 'rouge-2': 0.6, 'rouge-l': 0.75},
            'source_relevance': 0.9
        }

        # Ejecutar la función
        questions = ["Test question"]
        reference_answers = ["Test reference"]
        results = run_tests(mock_rag_system, questions, reference_answers)

        # Verificaciones
        mock_rag_system.ask_question.assert_called_once_with("Test question")
        mock_metrics_instance.evaluate_response.assert_called_once()
        mock_file.assert_called_once_with('test_results.csv', 'w', newline='', encoding='utf-8')
        mock_csv_writer.return_value.writeheader.assert_called_once()
        mock_csv_writer.return_value.writerow.assert_called_once()

        self.assertEqual(len(results), 1)
        self.assertIn('bleu_score', results[0])
        self.assertIn('rouge_scores_rouge-1', results[0])

    @patch('src.utils.test_runner.plt')
    def test_plot_results(self, mock_plt):
        results = [
            {'bleu_score': 0.8, 'rouge_scores_rouge-1': 0.7, 'rouge_scores_rouge-2': 0.6, 'rouge_scores_rouge-l': 0.75, 'source_relevance': 0.9, 'response_time': 1.0, 'question': 'Q1'},
            {'bleu_score': 0.7, 'rouge_scores_rouge-1': 0.6, 'rouge_scores_rouge-2': 0.5, 'rouge_scores_rouge-l': 0.65, 'source_relevance': 0.8, 'response_time': 1.5, 'question': 'Q2'}
        ]

        plot_results(results)

        # Verificar que se llamaron los métodos de matplotlib
        mock_plt.figure.assert_called_once()
        self.assertEqual(mock_plt.subplot.call_count, 2)
        self.assertEqual(mock_plt.plot.call_count, 7)  # 5 en el primer subplot, 2 en el segundo
        mock_plt.legend.assert_called()
        mock_plt.xticks.assert_called()
        mock_plt.tight_layout.assert_called_once()
        mock_plt.show.assert_called_once()

if __name__ == '__main__':
    unittest.main()