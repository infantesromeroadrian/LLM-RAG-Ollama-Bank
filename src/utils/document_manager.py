# 3.document_manager.py

"""
document_manager.py

Este módulo proporciona una clase para gestionar la carga y procesamiento de documentos.

Clases:
    DocumentManager: Gestiona la carga y procesamiento de documentos PDF y archivos CSV.

Dependencias:
    - src.utils.document_loader.DocumentLoader
    - src.utils.data_processor.DataProcessor
"""

from src.utils.document_loader import DocumentLoader
from src.utils.document_processor import DataProcessor

class DocumentManager:
    """
    Gestiona la carga y procesamiento de documentos PDF y archivos CSV.
    """

    def __init__(self, pdf_directory: str, csv_file: str):
        """
        Inicializa el DocumentManager con las rutas de los documentos.

        Args:
            pdf_directory (str): Ruta al directorio que contiene los archivos PDF.
            csv_file (str): Ruta al archivo CSV a procesar.
        """
        self.pdf_directory = pdf_directory
        self.csv_file = csv_file

    def load_and_process_documents(self):
        """
        Carga y procesa los documentos PDF y el archivo CSV.

        Returns:
            list: Una lista que contiene el resumen del CSV, los documentos
                  individuales del CSV y los documentos PDF divididos.
        """
        pdf_docs = DocumentLoader.load_pdfs(self.pdf_directory)
        df = DocumentLoader.load_csv(self.csv_file)

        split_docs = DataProcessor.split_documents(pdf_docs)
        csv_summary_doc = DataProcessor.create_csv_summary(df)
        csv_docs = DataProcessor.create_csv_docs(df)

        return [csv_summary_doc] + csv_docs + split_docs