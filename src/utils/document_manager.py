# 3.document_manager.py

"""
document_manager.py

Este módulo proporciona una clase DocumentManager que gestiona la carga y procesamiento
de documentos PDF y archivos CSV, utilizando las clases DocumentLoader y DataProcessor.

Dependencias:
- src.utils.document_loader
- src.utils.document_processor

Clases:
- DocumentManager
"""

from src.utils.document_loader import DocumentLoader
from src.utils.document_processor import DataProcessor


class DocumentManager:
    """
    Gestiona la carga y procesamiento de documentos PDF y archivos CSV.

    Esta clase coordina el uso de DocumentLoader para cargar documentos y
    DataProcessor para procesar los documentos cargados.

    Atributos:
        pdf_directory (str): Ruta al directorio que contiene los archivos PDF.
        csv_file (str): Ruta al archivo CSV a procesar.
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

        Este método realiza las siguientes operaciones:
        1. Carga los documentos PDF del directorio especificado.
        2. Carga el archivo CSV.
        3. Divide los documentos PDF.
        4. Crea un resumen del CSV.
        5. Crea documentos individuales a partir del CSV.
        6. Combina todos los documentos procesados en una sola lista.

        Returns:
            list: Una lista que contiene el resumen del CSV, los documentos
                  individuales del CSV y los documentos PDF divididos.
        """
        # Cargar documentos
        pdf_docs = DocumentLoader.load_pdfs(self.pdf_directory)
        df = DocumentLoader.load_csv(self.csv_file)

        # Procesar documentos
        split_docs = DataProcessor.split_documents(pdf_docs)
        csv_summary_doc = DataProcessor.create_csv_summary(df)
        csv_docs = DataProcessor.create_csv_docs(df)

        # Combinar y retornar todos los documentos procesados
        return [csv_summary_doc] + csv_docs + split_docs

# Ejemplo de uso:
# doc_manager = DocumentManager("/path/to/pdfs", "/path/to/data.csv")
# processed_docs = doc_manager.load_and_process_documents()