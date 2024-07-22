# test_vector_store_manager.py

import unittest
from unittest.mock import Mock, patch
import os
from src.features.vector_store_manager import VectorStoreManager
from langchain_core.documents import Document

class TestVectorStoreManager(unittest.TestCase):

    def setUp(self):
        self.mock_embed_model = Mock()
        self.base_dir = "/test/base/dir"
        self.vector_store_manager = VectorStoreManager(self.mock_embed_model, self.base_dir)

    @patch('src.utils.vector_store_manager.Chroma')
    def test_create_vector_store(self, mock_chroma):
        # Crear documentos de prueba
        test_documents = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"})
        ]

        # Configurar el mock de Chroma
        mock_chroma_instance = Mock()
        mock_chroma.from_documents.return_value = mock_chroma_instance

        # Llamar al método create_vector_store
        result = self.vector_store_manager.create_vector_store(test_documents)

        # Verificar que Chroma.from_documents fue llamado con los argumentos correctos
        expected_persist_directory = os.path.join(self.base_dir, "bank_data_db")
        mock_chroma.from_documents.assert_called_once_with(
            documents=test_documents,
            embedding=self.mock_embed_model,
            persist_directory=expected_persist_directory,
            collection_name="bank_regulations_and_data"
        )

        # Verificar que el método devuelve la instancia de Chroma
        self.assertEqual(result, mock_chroma_instance)

    def test_init(self):
        # Verificar que el constructor inicializa correctamente los atributos
        self.assertEqual(self.vector_store_manager.embed_model, self.mock_embed_model)
        self.assertEqual(self.vector_store_manager.base_dir, self.base_dir)

    @patch('os.path.join')
    def test_persist_directory_creation(self, mock_join):
        # Configurar el mock de os.path.join
        expected_path = "/test/base/dir/bank_data_db"
        mock_join.return_value = expected_path

        # Crear documentos de prueba
        test_documents = [Document(page_content="Test document", metadata={"source": "test"})]

        # Configurar el mock de Chroma (necesario aunque no lo usemos directamente en esta prueba)
        with patch('src.utils.vector_store_manager.Chroma') as mock_chroma:
            self.vector_store_manager.create_vector_store(test_documents)

        # Verificar que os.path.join fue llamado con los argumentos correctos
        mock_join.assert_called_once_with(self.base_dir, "bank_data_db")

    @patch('src.utils.vector_store_manager.Chroma')
    def test_create_vector_store_empty_documents(self, mock_chroma):
        # Probar con una lista vacía de documentos
        empty_documents = []

        # Llamar al método create_vector_store
        self.vector_store_manager.create_vector_store(empty_documents)

        # Verificar que Chroma.from_documents fue llamado con una lista vacía
        mock_chroma.from_documents.assert_called_once()
        call_args = mock_chroma.from_documents.call_args
        self.assertEqual(len(call_args[1]['documents']), 0)

if __name__ == '__main__':
    unittest.main()