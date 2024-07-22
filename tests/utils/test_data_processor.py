# test_data_processor.py

import unittest
import pandas as pd
import numpy as np
from langchain_core.documents import Document
from src.utils.document_processor import DataProcessor
import json

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        # Crear un DataFrame de prueba
        self.df = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'age': [25, 30, 35, 40, 45],
            'balance': [1000.5, 2000.75, 1500.25, 3000.0, 2500.5],
            'country': ['USA', 'UK', 'USA', 'Canada', 'UK']
        })

        # Crear una lista de documentos de prueba
        self.docs = [
            Document(page_content="This is the first document."),
            Document(page_content="This is the second document with more content."),
            Document(page_content="And this is the third one.")
        ]

    def test_split_documents(self):
        split_docs = DataProcessor.split_documents(self.docs, chunk_size=10, chunk_overlap=2)
        self.assertGreater(len(split_docs), len(self.docs))
        self.assertIsInstance(split_docs[0], Document)

    def test_create_csv_summary(self):
        summary_doc = DataProcessor.create_csv_summary(self.df)
        self.assertIsInstance(summary_doc, Document)
        self.assertIn("RESUMEN DETALLADO DEL CSV", summary_doc.page_content)
        self.assertIn("Total de filas: 5", summary_doc.page_content)
        self.assertIn("Clientes únicos: 5", summary_doc.page_content)

        # Verificar que el JSON en los metadatos sea válido
        full_summary = json.loads(summary_doc.metadata['full_summary'])
        self.assertIn('general_info', full_summary)
        self.assertIn('numerical_stats', full_summary)
        self.assertIn('categorical_stats', full_summary)
        self.assertIn('correlations', full_summary)

    def test_create_csv_docs(self):
        csv_docs = DataProcessor.create_csv_docs(self.df)
        self.assertEqual(len(csv_docs), len(self.df))
        self.assertIsInstance(csv_docs[0], Document)
        self.assertEqual(csv_docs[0].metadata['customer_id'], 1)

    def test_create_csv_docs_with_sample(self):
        sample_size = 3
        csv_docs = DataProcessor.create_csv_docs(self.df, sample_size=sample_size)
        self.assertEqual(len(csv_docs), sample_size)

    def test_get_customer_by_id(self):
        csv_docs = DataProcessor.create_csv_docs(self.df)
        customer_doc = DataProcessor.get_customer_by_id(csv_docs, 3)
        self.assertIsNotNone(customer_doc)
        self.assertEqual(customer_doc.metadata['customer_id'], 3)

    def test_get_customer_by_id_not_found(self):
        csv_docs = DataProcessor.create_csv_docs(self.df)
        customer_doc = DataProcessor.get_customer_by_id(csv_docs, 999)  # ID que no existe
        self.assertIsNone(customer_doc)

    def test_serialize_numpy(self):
        # Probar la función serialize_numpy indirectamente a través de create_csv_summary
        df_with_numpy = self.df.copy()
        df_with_numpy['numpy_int'] = np.int64(10)
        df_with_numpy['numpy_float'] = np.float64(20.5)
        summary_doc = DataProcessor.create_csv_summary(df_with_numpy)
        full_summary = json.loads(summary_doc.metadata['full_summary'])
        self.assertIsInstance(full_summary['numerical_stats']['numpy_int']['min'], int)
        self.assertIsInstance(full_summary['numerical_stats']['numpy_float']['min'], float)

if __name__ == '__main__':
    unittest.main()