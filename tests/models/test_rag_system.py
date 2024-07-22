# test_rag_system.py

import unittest
from unittest.mock import Mock, patch
from src.models.rag_system import RAGSystem

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.pdf_directory = "/test/pdfs"
        self.csv_file = "/test/data.csv"
        self.base_dir = "/test/base"
        self.rag_system = RAGSystem(self.pdf_directory, self.csv_file, self.base_dir)

    def test_init(self):
        self.assertEqual(self.rag_system.pdf_directory, self.pdf_directory)
        self.assertEqual(self.rag_system.csv_file, self.csv_file)
        self.assertIsInstance(self.rag_system.llm, Mock)  # Asumiendo que Ollama es un mock
        self.assertIsInstance(self.rag_system.embed_model, Mock)  # Asumiendo que FastEmbedEmbeddings es un mock
        self.assertIsInstance(self.rag_system.vector_store_manager, Mock)
        self.assertIsNone(self.rag_system.qa_system)

    @patch('src.models.rag_system.DocumentLoader')
    @patch('src.models.rag_system.DataProcessor')
    @patch('src.models.rag_system.CustomRetriever')
    @patch('src.models.rag_system.QASystem')
    def test_run(self, mock_qa_system, mock_custom_retriever, mock_data_processor, mock_document_loader):
        # Configurar mocks
        mock_document_loader.load_pdfs.return_value = ["pdf1", "pdf2"]
        mock_document_loader.load_csv.return_value = Mock()
        mock_data_processor.split_documents.return_value = ["split1", "split2"]
        mock_data_processor.create_csv_summary.return_value = "csv_summary"
        mock_data_processor.create_csv_docs.return_value = ["csv1", "csv2"]
        mock_vector_store = Mock()
        self.rag_system.vector_store_manager.create_vector_store.return_value = mock_vector_store
        mock_vector_store._collection.count.return_value = 5

        # Ejecutar el método run
        self.rag_system.run()

        # Verificar que se llamaron los métodos correctos
        mock_document_loader.load_pdfs.assert_called_once_with(self.pdf_directory)
        mock_document_loader.load_csv.assert_called_once_with(self.csv_file)
        mock_data_processor.split_documents.assert_called_once()
        mock_data_processor.create_csv_summary.assert_called_once()
        mock_data_processor.create_csv_docs.assert_called_once()
        self.rag_system.vector_store_manager.create_vector_store.assert_called_once()
        mock_custom_retriever.assert_called_once_with(vectorstore=mock_vector_store)
        mock_qa_system.assert_called_once()
        mock_qa_system.return_value.setup_qa_chain.assert_called_once()

    def test_ask_question_without_run(self):
        with self.assertRaises(ValueError):
            self.rag_system.ask_question("Test question")

    @patch('src.models.rag_system.QASystem')
    def test_ask_question(self, mock_qa_system):
        # Configurar el sistema QA mock
        mock_qa = Mock()
        mock_qa.ask_question.return_value = {"result": "Test answer", "source_documents": ["doc1"]}
        self.rag_system.qa_system = mock_qa

        # Realizar una pregunta
        result = self.rag_system.ask_question("Test question")

        # Verificar el resultado
        self.assertEqual(result["result"], "Test answer")
        self.assertEqual(result["source_documents"], ["doc1"])
        mock_qa.ask_question.assert_called_once_with("Test question")

    def test_get_system_info(self):
        info = self.rag_system.get_system_info()
        self.assertEqual(info["pdf_directory"], self.pdf_directory)
        self.assertEqual(info["csv_file"], self.csv_file)
        self.assertEqual(info["llm_model"], "llama3")
        self.assertEqual(info["embedding_model"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(info["vector_store_base_dir"], self.base_dir)
        self.assertEqual(info["qa_system_status"], "Not configured")

if __name__ == '__main__':
    unittest.main()