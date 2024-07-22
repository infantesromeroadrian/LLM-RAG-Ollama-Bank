# test_app.py


import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import plotly.graph_objects as go
from src.app import StreamlitRAGSystem, main

class TestApp(unittest.TestCase):

    @patch('src.app.RAGSystem.__init__')
    @patch('src.app.Ollama')
    def test_streamlit_rag_system_init(self, mock_ollama, mock_rag_init):
        mock_rag_init.return_value = None
        system = StreamlitRAGSystem("base_dir")
        self.assertEqual(system.model_name, "llama3")
        self.assertEqual(system.temperature, 0.7)
        self.assertEqual(system.chunk_size, 2000)
        self.assertEqual(system.chunk_overlap, 500)
        mock_ollama.assert_called_once_with(model="llama3", temperature=0.7)

    @patch('src.app.RAGSystem.__init__')
    @patch('src.app.Ollama')
    def test_update_parameters(self, mock_ollama, mock_rag_init):
        mock_rag_init.return_value = None
        system = StreamlitRAGSystem("base_dir")
        system.run = MagicMock()
        system.update_parameters("llama2", 0.5, 1500, 300)
        self.assertEqual(system.model_name, "llama2")
        self.assertEqual(system.temperature, 0.5)
        self.assertEqual(system.chunk_size, 1500)
        self.assertEqual(system.chunk_overlap, 300)
        mock_ollama.assert_called_with(model="llama2", temperature=0.5)
        system.run.assert_called_once()

    @patch('src.app.st')
    @patch('src.app.StreamlitRAGSystem')
    @patch('src.app.os.path.exists')
    @patch('src.app.pd.read_csv')
    @patch('src.app.make_subplots')
    @patch('src.app.go.Bar')
    @patch('src.app.go.Scatter')
    def test_main_function(self, mock_scatter, mock_bar, mock_subplots, mock_read_csv, mock_exists, mock_streamlit_rag, mock_st):
        # Simular la carga del sistema RAG
        mock_rag_instance = MagicMock()
        mock_streamlit_rag.return_value = mock_rag_instance
        mock_st.cache_resource.return_value = lambda: mock_rag_instance

        # Simular la entrada del usuario y la ejecución de pruebas
        mock_st.text_input.return_value = "Test question"
        mock_st.button.side_effect = [True, True]  # Simular clic en "Obtener Respuesta" y "Ejecutar Pruebas"
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({
            'bleu_score': [0.8], 'rouge_scores_rouge-1': [0.7],
            'rouge_scores_rouge-2': [0.6], 'rouge_scores_rouge-l': [0.75],
            'response_time': [1.0], 'source_relevance': [0.9]
        })

        # Simular la respuesta del sistema RAG
        mock_rag_instance.ask_question.return_value = {
            "result": "Test answer",
            "source_documents": [MagicMock(metadata={"source": "Test source"}, page_content="Test content")]
        }

        # Ejecutar la función main
        main()

        # Verificar las llamadas a Streamlit
        mock_st.title.assert_called_once()
        mock_st.sidebar.selectbox.assert_called_once()
        mock_st.sidebar.slider.assert_called_once()
        mock_st.sidebar.number_input.assert_called()
        mock_st.text_input.assert_called_once()
        mock_st.button.assert_called()
        mock_st.write.assert_called()
        mock_st.plotly_chart.assert_called_once()

        # Verificar la creación de gráficos
        mock_subplots.assert_called_once()
        mock_bar.assert_called()
        mock_scatter.assert_called()

if __name__ == '__main__':
    unittest.main()