# test_document_manager.py

import unittest
from unittest.mock import patch, MagicMock
from src.utils.document_manager import DocumentManager
from langchain_core.documents import Document
import pandas as pd

class TestDocumentManager(unittest.TestCase):

    def setUp(self):
        self.pdf_directory = "/path/to/pdfs"
        self.csv_file = "/path/to/csv/file.csv"
        self.document_manager = DocumentManager(self.pdf_directory, self.csv_file)

    def test_init(self):
        self.assertEqual(self.document_manager.pdf_directory, self.pdf_directory)
        self.assertEqual(self.document_manager.csv_file, self.csv_file)

    @patch('src.utils.document_manager.DocumentLoader')
    @patch('src.utils.document_manager.DataProcessor')
    def test_load_and_process_documents(self, mock_data_processor, mock_document_loader):
        # Configurar mocks
        mock_pdf_docs = [Document(page_content="PDF1"), Document(page_content="PDF2")]
        mock_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_split_docs = [Document(page_content="Split1"), Document(page_content="Split2")]
        mock_csv_summary_doc = Document(page_content="CSV Summary")
        mock_csv_docs = [Document(page_content="CSV1"), Document(page_content="CSV2")]

        mock_document_loader.load_pdfs.return_value = mock_pdf_docs
        mock_document_loader.load_csv.return_value = mock_df
        mock_data_processor.split_documents.return_value = mock_split_docs
        mock_data_processor.create_csv_summary.return_value = mock_csv_summary_doc
        mock_data_processor.create_csv_docs.return_value = mock_csv_docs

        # Llamar al método
        result = self.document_manager.load_and_process_documents()

        # Verificar llamadas a métodos mock
        mock_document_loader.load_pdfs.assert_called_once_with(self.pdf_directory)
        mock_document_loader.load_csv.assert_called_once_with(self.csv_file)
        mock_data_processor.split_documents.assert_called_once_with(mock_pdf_docs)
        mock_data_processor.create_csv_summary.assert_called_once_with(mock_df)
        mock_data_processor.create_csv_docs.assert_called_once_with(mock_df)

        # Verificar resultado
        expected_result = [mock_csv_summary_doc] + mock_csv_docs + mock_split_docs
        self.assertEqual(result, expected_result)

    @patch('src.utils.document_manager.DocumentLoader')
    @patch('src.utils.document_manager.DataProcessor')
    def test_load_and_process_documents_empty(self, mock_data_processor, mock_document_loader):
        # Configurar mocks para simular documentos vacíos
        mock_document_loader.load_pdfs.return_value = []
        mock_document_loader.load_csv.return_value = pd.DataFrame()
        mock_data_processor.split_documents.return_value = []
        mock_data_processor.create_csv_summary.return_value = Document(page_content="Empty CSV Summary")
        mock_data_processor.create_csv_docs.return_value = []

        # Llamar al método
        result = self.document_manager.load_and_process_documents()

        # Verificar resultado
        self.assertEqual(len(result), 1)  # Solo debería contener el resumen del CSV vacío
        self.assertEqual(result[0].page_content, "Empty CSV Summary")

    @patch('src.utils.document_manager.DocumentLoader')
    @patch('src.utils.document_manager.DataProcessor')
    def test_load_and_process_documents_error_handling(self, mock_data_processor, mock_document_loader):
        # Simular un error al cargar PDFs
        mock_document_loader.load_pdfs.side_effect = Exception("Error loading PDFs")

        # Verificar que se maneja la excepción
        with self.assertRaises(Exception):
            self.document_manager.load_and_process_documents()

if __name__ == '__main__':
    unittest.main()