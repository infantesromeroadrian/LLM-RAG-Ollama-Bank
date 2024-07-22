# test_document_loader.py

import unittest
import os
import tempfile
import pandas as pd
from src.utils.document_loader import DocumentLoader
from langchain_core.documents import Document

class TestDocumentLoader(unittest.TestCase):

    def setUp(self):
        # Crear un directorio temporal para las pruebas
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Limpiar el directorio temporal después de las pruebas
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_load_pdfs_empty_directory(self):
        # Probar carga de PDFs de un directorio vacío
        docs = DocumentLoader.load_pdfs(self.test_dir)
        self.assertEqual(len(docs), 0)

    def test_load_pdfs_with_files(self):
        # Crear archivos PDF de prueba
        pdf_content = b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%EOF"
        with open(os.path.join(self.test_dir, "test1.pdf"), "wb") as f:
            f.write(pdf_content)
        with open(os.path.join(self.test_dir, "test2.pdf"), "wb") as f:
            f.write(pdf_content)

        docs = DocumentLoader.load_pdfs(self.test_dir)
        self.assertEqual(len(docs), 2)
        self.assertIsInstance(docs[0], Document)

    def test_load_pdfs_ignore_non_pdf(self):
        # Crear un archivo que no es PDF
        with open(os.path.join(self.test_dir, "not_a_pdf.txt"), "w") as f:
            f.write("This is not a PDF")

        docs = DocumentLoader.load_pdfs(self.test_dir)
        self.assertEqual(len(docs), 0)

    def test_load_csv_file_exists(self):
        # Crear un archivo CSV de prueba
        csv_content = "id,name\n1,test1\n2,test2"
        csv_path = os.path.join(self.test_dir, "test.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)

        df = DocumentLoader.load_csv(csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["id", "name"])

    def test_load_csv_file_not_found(self):
        # Intentar cargar un archivo CSV que no existe
        with self.assertRaises(FileNotFoundError):
            DocumentLoader.load_csv("non_existent_file.csv")

if __name__ == '__main__':
    unittest.main()