# test_custom_retriever.py

import unittest
from unittest.mock import Mock, AsyncMock
import asyncio
from src.features.custom_retriever import CustomRetriever
from langchain_core.documents import Document

class TestCustomRetriever(unittest.TestCase):

    def setUp(self):
        self.mock_vectorstore = Mock()
        self.retriever = CustomRetriever(vectorstore=self.mock_vectorstore)

    def test_get_relevant_documents(self):
        # Crear documentos de prueba
        csv_summary_doc = Document(page_content="CSV Summary", metadata={"source": "CSV_summary"})
        other_docs = [
            Document(page_content="Doc 1", metadata={"source": "other"}),
            Document(page_content="Doc 2", metadata={"source": "other"}),
            Document(page_content="Doc 3", metadata={"source": "other"}),
            Document(page_content="Doc 4", metadata={"source": "other"})
        ]

        # Configurar el comportamiento del mock
        self.mock_vectorstore.similarity_search.side_effect = [
            [csv_summary_doc],  # Para la búsqueda del resumen CSV
            other_docs  # Para la búsqueda de otros documentos
        ]

        # Ejecutar el método y verificar los resultados
        query = "test query"
        result = self.retriever.get_relevant_documents(query)

        # Verificar que se llamó a similarity_search dos veces con los argumentos correctos
        self.mock_vectorstore.similarity_search.assert_any_call(query, filter={"source": "CSV_summary"}, k=1)
        self.mock_vectorstore.similarity_search.assert_any_call(query, k=4)

        # Verificar que el resultado tiene la estructura esperada
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], csv_summary_doc)
        self.assertEqual(result[1:], other_docs)

    def test_get_relevant_documents_no_csv_summary(self):
        # Simular que no se encuentra un resumen CSV
        other_docs = [
            Document(page_content="Doc 1", metadata={"source": "other"}),
            Document(page_content="Doc 2", metadata={"source": "other"}),
            Document(page_content="Doc 3", metadata={"source": "other"}),
            Document(page_content="Doc 4", metadata={"source": "other"})
        ]

        self.mock_vectorstore.similarity_search.side_effect = [
            [],  # No se encuentra resumen CSV
            other_docs
        ]

        query = "test query"
        result = self.retriever.get_relevant_documents(query)

        self.assertEqual(len(result), 4)
        self.assertEqual(result, other_docs)

    def test_async_get_relevant_documents(self):
        # Crear documentos de prueba
        csv_summary_doc = Document(page_content="CSV Summary", metadata={"source": "CSV_summary"})
        other_docs = [
            Document(page_content="Doc 1", metadata={"source": "other"}),
            Document(page_content="Doc 2", metadata={"source": "other"}),
            Document(page_content="Doc 3", metadata={"source": "other"}),
            Document(page_content="Doc 4", metadata={"source": "other"})
        ]

        # Configurar el comportamiento del mock
        self.mock_vectorstore.similarity_search.side_effect = [
            [csv_summary_doc],
            other_docs
        ]

        # Ejecutar el método asíncrono
        query = "test query"
        result = asyncio.run(self.retriever.aget_relevant_documents(query))

        # Verificar que el resultado es el mismo que en la versión síncrona
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], csv_summary_doc)
        self.assertEqual(result[1:], other_docs)

if __name__ == '__main__':
    unittest.main()