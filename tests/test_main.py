# test_main.py

import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
from src.main import main

class TestMain(unittest.TestCase):

    @patch('src.main.RAGSystem')
    @patch('src.main.run_tests')
    @patch('src.main.plot_results')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_main_function(self, mock_print, mock_input, mock_plot_results, mock_run_tests, mock_rag_system):
        # Configurar el mock para RAGSystem
        mock_rag_instance = mock_rag_system.return_value
        mock_rag_instance.ask_question.return_value = {"result": "Respuesta de prueba"}

        # Configurar el mock para run_tests
        mock_run_tests.return_value = [{"metric1": 0.8, "metric2": 0.9}]

        # Simular entradas del usuario
        mock_input.side_effect = ["¿Pregunta de prueba?", "salir"]

        # Ejecutar la función main
        main()

        # Verificar que RAGSystem se inicializó correctamente
        mock_rag_system.assert_called_once_with(
            pdf_directory="../data/GuideLines",
            csv_file="../data/raw_data/BankCustomerChurnPrediction.csv",
            base_dir="../data"
        )
        mock_rag_instance.run.assert_called_once()

        # Verificar que run_tests fue llamado con los argumentos correctos
        mock_run_tests.assert_called_once()
        args, _ = mock_run_tests.call_args
        self.assertEqual(args[0], mock_rag_instance)
        self.assertEqual(len(args[1]), 4)  # 4 preguntas de prueba
        self.assertEqual(len(args[2]), 4)  # 4 respuestas de referencia
        self.assertEqual(args[3], "resultados_test.csv")

        # Verificar que plot_results fue llamado
        mock_plot_results.assert_called_once()

        # Verificar el bucle interactivo
        mock_rag_instance.ask_question.assert_called_once_with("¿Pregunta de prueba?")
        mock_print.assert_called_with("Respuesta: Respuesta de prueba")

    @patch('src.main.RAGSystem')
    @patch('src.main.run_tests')
    @patch('src.main.plot_results')
    @patch('builtins.input')
    def test_main_function_error_handling(self, mock_input, mock_plot_results, mock_run_tests, mock_rag_system):
        # Simular un error en RAGSystem
        mock_rag_instance = mock_rag_system.return_value
        mock_rag_instance.run.side_effect = Exception("Error de prueba")

        # Ejecutar la función main y verificar que maneja la excepción
        with self.assertRaises(Exception):
            main()

    @patch('src.main.RAGSystem')
    @patch('src.main.run_tests')
    @patch('src.main.plot_results')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_main_function_empty_input(self, mock_print, mock_input, mock_plot_results, mock_run_tests, mock_rag_system):
        # Configurar el mock para RAGSystem
        mock_rag_instance = mock_rag_system.return_value

        # Simular entradas vacías del usuario seguidas de 'salir'
        mock_input.side_effect = ["", "  ", "salir"]

        # Ejecutar la función main
        main()

        # Verificar que no se llamó a ask_question con entradas vacías
        mock_rag_instance.ask_question.assert_not_called()

if __name__ == '__main__':
    unittest.main()